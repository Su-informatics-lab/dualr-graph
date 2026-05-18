#!/usr/bin/env python3
"""
train.py — Semi-supervised 5-fold CV for the feature-graph autoencoder.

Implements Dr. Su's whiteboard objective:

    L = lambda_rec * L_rec + L_AKI

Defaults match the whiteboard sketch:
  - AKI prediction = reconstruction of AKI node (not separate head)
  - Train AKI labels visible as input; val/test AKI masked
  - Reconstruction loss uses ALL patients (transductive, semi-supervised)
  - lambda_rec = 0.5 (tunable toward 0)
  - Denoising masking (20%) prevents identity shortcut
  - Missing values: iterative imputation from model predictions

Quick start:
    python train.py --edge_method llm --use_edge_weights --layers 3

First sweep:
    for E in llm spearman mi; do
      for L in 2 3; do
        for R in 0 0.1 0.5 1.0; do
          python train.py --edge_method $E --layers $L --lambda_rec $R \\
            --run_name gae_${E}_l${L}_lam${R}
        done
      done
    done
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from graph import build_mi, build_spearman, knn_sparsify, load_llm_graph
from models import build_model

# ──────────────────────────────────────────────────────────────
#  Fold data container
# ──────────────────────────────────────────────────────────────


class FoldData:
    __slots__ = (
        "x_filled",
        "x_target",
        "observed_mask",
        "input_missing",
        "y",
        "train_prior",
        "stats",
    )

    def __init__(
        self, x_filled, x_target, observed_mask, input_missing, y, train_prior, stats
    ):
        self.x_filled = x_filled  # [N, F]
        self.x_target = x_target  # [N, F]
        self.observed_mask = observed_mask  # [N, F] bool
        self.input_missing = input_missing  # [N, F] bool
        self.y = y  # [N]
        self.train_prior = train_prior  # float
        self.stats = stats


# ──────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auroc(y_true, y_score):
    return (
        float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )


def safe_auprc(y_true, y_score):
    return (
        float(average_precision_score(y_true, y_score))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )


def iter_batches(indices, batch_size, shuffle=True):
    indices = np.asarray(indices)
    if shuffle:
        indices = np.random.permutation(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _mode_or_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    vals, counts = torch.unique(values.float(), return_counts=True)
    return float(vals[counts.argmax()].item())


# ──────────────────────────────────────────────────────────────
#  Per-fold preprocessing
# ──────────────────────────────────────────────────────────────


def preprocess_fold(
    X_nf: torch.Tensor,
    missing_nf: torch.Tensor,
    binary_mask_f: torch.Tensor,
    train_idx: np.ndarray,
    aki_idx: int,
    impute_strategy: str = "mean",
    use_label_input: bool = True,
) -> FoldData:
    """
    Standardize continuous features, impute missing, handle AKI column.

    Default (use_label_input=True, whiteboard design):
      - Train patients: AKI column = true label, missing_flag = False
      - Val/test patients: AKI column = train prior, missing_flag = True

    Ablation (use_label_input=False):
      - All patients: AKI column = train prior, missing_flag = True
      - Labels enter only through L_AKI
    """
    X_nf = X_nf.float().clone()
    missing_nf = missing_nf.bool().clone()
    binary_mask_f = binary_mask_f.bool().clone()
    N, F = X_nf.shape
    train_t = torch.as_tensor(train_idx, dtype=torch.long)

    y = X_nf[:, aki_idx].float().clone()
    prior = float(y[train_t].mean().item())
    if not np.isfinite(prior):
        prior = 0.0

    x_filled = X_nf.clone()
    x_target = X_nf.clone()
    observed = ~missing_nf.clone()
    input_missing = missing_nf.clone()
    stats: Dict = {}

    for j in range(F):
        if j == aki_idx:
            continue

        obs_train = train_t[~missing_nf[train_t, j]]
        vals_train = X_nf[obs_train, j].float()

        if binary_mask_f[j]:
            fill = _mode_or_mean(vals_train)
            x_filled[missing_nf[:, j], j] = fill
            x_target[missing_nf[:, j], j] = fill
            stats[str(j)] = {"kind": "binary", "fill": fill}
        else:
            if vals_train.numel() == 0:
                mean, std, fill_raw = 0.0, 1.0, 0.0
            else:
                mean = float(vals_train.mean().item())
                std = float(vals_train.std(unbiased=False).item())
                if std < 1e-6:
                    std = 1.0
                fill_raw = (
                    float(vals_train.median().item())
                    if impute_strategy == "median"
                    else mean
                )

            obs_all = ~missing_nf[:, j]
            x_filled[obs_all, j] = (X_nf[obs_all, j] - mean) / std
            x_target[obs_all, j] = x_filled[obs_all, j]
            fill_std = (fill_raw - mean) / std
            x_filled[missing_nf[:, j], j] = fill_std
            x_target[missing_nf[:, j], j] = fill_std
            stats[str(j)] = {
                "kind": "continuous",
                "mean": mean,
                "std": std,
                "fill": fill_std,
            }

    # ── AKI column ────────────────────────────────────────────
    if use_label_input:
        # Whiteboard: train labels visible, val/test masked
        x_filled[:, aki_idx] = prior
        x_filled[train_t, aki_idx] = y[train_t]
        input_missing[:, aki_idx] = True
        input_missing[train_t, aki_idx] = False
    else:
        # Ablation: all patients get prior, labels only in L_AKI
        x_filled[:, aki_idx] = prior
        input_missing[:, aki_idx] = True

    x_target[:, aki_idx] = prior
    observed[:, aki_idx] = False

    return FoldData(
        x_filled=x_filled,
        x_target=x_target,
        observed_mask=observed,
        input_missing=input_missing,
        y=y,
        train_prior=prior,
        stats=stats,
    )


# ──────────────────────────────────────────────────────────────
#  Graph construction (per-fold for data-derived edges)
# ──────────────────────────────────────────────────────────────


def make_graph_for_fold(config, fold_data, train_idx, aki_idx, data_dir):
    """Build feature graph without using val/test AKI labels."""
    method = config["edge_method"]
    if method in ("spearman", "mi"):
        x_graph = fold_data.x_filled.clone()
        x_graph[:, aki_idx] = fold_data.train_prior
        x_graph[train_idx, aki_idx] = fold_data.y[train_idx]
        X_train_np = x_graph[train_idx].cpu().numpy()
        if method == "spearman":
            adj = build_spearman(X_train_np)
        else:
            adj = build_mi(X_train_np)
        edge_index, edge_weight = knn_sparsify(adj, k=config["k"], directed=False)
    else:
        edge_index, edge_weight = load_llm_graph(
            data_dir=data_dir,
            k=config["k"],
            cv_cutoff=config.get("cv_cutoff"),
        )
    return edge_index.long(), edge_weight.float()


# ──────────────────────────────────────────────────────────────
#  Denoising masking
# ──────────────────────────────────────────────────────────────


def denoise_batch(
    x_filled, input_missing, observed_mask, batch_idx, non_aki_mask_f, mask_rate
):
    """Randomly hide observed non-AKI cells (denoising autoencoder)."""
    idx = torch.as_tensor(batch_idx, dtype=torch.long, device=x_filled.device)
    x_b = x_filled[idx].clone()
    miss_b = input_missing[idx].clone()
    if mask_rate > 0:
        eligible = observed_mask[idx] & non_aki_mask_f.unsqueeze(0)
        drop = (torch.rand_like(x_b) < mask_rate) & eligible
        x_b[drop] = 0.0
        miss_b = miss_b | drop
    return x_b, miss_b


# ──────────────────────────────────────────────────────────────
#  Loss
# ──────────────────────────────────────────────────────────────


def compute_loss(
    out,
    batch_idx,
    fold_data,
    train_indicator,
    non_aki_mask_f,
    lambda_rec,
    pseudo_weight,
    pos_weight,
):
    device = out.x_hat.device
    idx = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
    target = fold_data.x_target[idx].to(device)
    observed = fold_data.observed_mask[idx].to(device)

    non_aki = non_aki_mask_f.to(device).unsqueeze(0)
    rec_w = (observed & non_aki).float()
    if pseudo_weight > 0:
        rec_w = rec_w + pseudo_weight * ((~observed) & non_aki).float()

    if lambda_rec > 0 and rec_w.sum() > 0:
        rec_loss = ((out.x_hat - target).pow(2) * rec_w).sum() / rec_w.sum().clamp_min(
            1.0
        )
    else:
        rec_loss = torch.tensor(0.0, device=device)

    ce_mask = train_indicator[idx]
    if ce_mask.any():
        y = fold_data.y[idx].to(device)
        aki_loss = F.binary_cross_entropy_with_logits(
            out.aki_logits[ce_mask],
            y[ce_mask],
            pos_weight=pos_weight,
        )
    else:
        aki_loss = torch.tensor(0.0, device=device)

    loss = lambda_rec * rec_loss + aki_loss
    return loss, {"rec": float(rec_loss.detach()), "aki": float(aki_loss.detach())}


# ──────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def score_indices(
    model, x_filled, input_missing, edge_index, edge_attr, indices, batch_size
):
    model.eval()
    probs = []
    for batch_idx in iter_batches(indices, batch_size=batch_size, shuffle=False):
        idx = torch.as_tensor(batch_idx, dtype=torch.long, device=x_filled.device)
        out = model(
            x_filled[idx],
            edge_index=edge_index,
            edge_attr=edge_attr,
            missing_mask=input_missing[idx],
        )
        probs.append(torch.sigmoid(out.aki_logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


# ──────────────────────────────────────────────────────────────
#  Iterative imputation
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def update_missing_from_recon(
    model, fold_data, edge_index, edge_attr, non_aki_mask_f, batch_size
):
    """Replace missing non-AKI cells with detached reconstruction."""
    model.eval()
    N = fold_data.x_filled.size(0)
    dev = fold_data.x_filled.device
    missing_non_aki = (~fold_data.observed_mask) & non_aki_mask_f.unsqueeze(0).to(dev)
    for batch_idx in iter_batches(np.arange(N), batch_size=batch_size, shuffle=False):
        idx = torch.as_tensor(batch_idx, dtype=torch.long, device=dev)
        out = model(
            fold_data.x_filled[idx],
            edge_index=edge_index,
            edge_attr=edge_attr,
            missing_mask=fold_data.input_missing[idx],
        )
        m = missing_non_aki[idx]
        if m.any():
            new_vals = out.x_hat.detach()
            fold_data.x_filled[idx] = torch.where(m, new_vals, fold_data.x_filled[idx])
            fold_data.x_target[idx] = torch.where(m, new_vals, fold_data.x_target[idx])


# ──────────────────────────────────────────────────────────────
#  Single fold
# ──────────────────────────────────────────────────────────────


def train_one_fold(
    fold,
    config,
    X_nf,
    missing_nf,
    binary_mask_f,
    feature_names,
    train_idx,
    val_idx,
    test_idx,
    data_dir,
    device,
):
    aki_idx = feature_names.index("aki_event")
    num_features = X_nf.shape[1]
    non_aki_mask_f = torch.ones(num_features, dtype=torch.bool, device=device)
    non_aki_mask_f[aki_idx] = False

    fold_data = preprocess_fold(
        X_nf,
        missing_nf,
        binary_mask_f,
        train_idx,
        aki_idx,
        impute_strategy=config["impute_strategy"],
        use_label_input=config["use_label_input"],
    )
    fold_data.x_filled = fold_data.x_filled.to(device)
    fold_data.x_target = fold_data.x_target.to(device)
    fold_data.observed_mask = fold_data.observed_mask.to(device)
    fold_data.input_missing = fold_data.input_missing.to(device)
    fold_data.y = fold_data.y.to(device)

    edge_index, edge_weight = make_graph_for_fold(
        config, fold_data, train_idx, aki_idx, data_dir
    )
    edge_index = edge_index.to(device)
    use_ew = config.get("use_edge_weights", False)
    edge_attr = edge_weight.unsqueeze(-1).to(device) if use_ew else None
    edge_dim = 1 if use_ew else None

    model = build_model(
        num_features=num_features,
        aki_idx=aki_idx,
        node_emb_dim=config["node_emb_dim"],
        hidden=config["hidden"],
        latent=config["latent"],
        heads=config["heads"],
        layers=config["layers"],
        dropout=config["dropout"],
        edge_dim=edge_dim,
        conv_type=config["conv_type"],
        use_recon_aki=config["use_recon_aki"],
        add_self_loops=config["add_self_loops"],
    ).to(device)

    y_train = fold_data.y[torch.as_tensor(train_idx, device=device)]
    n_pos = (y_train == 1).sum().float().clamp_min(1.0)
    n_neg = (y_train == 0).sum().float().clamp_min(1.0)
    pos_weight = (n_neg / n_pos).clamp(max=config["max_pos_weight"])

    train_indicator = torch.zeros(X_nf.shape[0], dtype=torch.bool, device=device)
    train_indicator[torch.as_tensor(train_idx, device=device)] = True

    # Transductive (default): reconstruct all patients.
    # Strict: only train patients contribute gradients.
    fit_idx = np.arange(X_nf.shape[0]) if not config["strict_recon"] else train_idx

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    best_val = -np.inf
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    patience_left = config["patience"]
    history: List[Dict] = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_losses = []
        for batch_idx in iter_batches(
            fit_idx, batch_size=config["batch_size"], shuffle=True
        ):
            x_b, miss_b = denoise_batch(
                fold_data.x_filled,
                fold_data.input_missing,
                fold_data.observed_mask,
                batch_idx,
                non_aki_mask_f,
                config["mask_rate"],
            )
            out = model(
                x_b, edge_index=edge_index, edge_attr=edge_attr, missing_mask=miss_b
            )
            loss, parts = compute_loss(
                out,
                batch_idx,
                fold_data,
                train_indicator,
                non_aki_mask_f,
                config["lambda_rec"],
                config["pseudo_weight"],
                pos_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss fold={fold} epoch={epoch}")
            if loss.requires_grad and loss.item() != 0.0:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
            epoch_losses.append({"loss": float(loss.detach()), **parts})

        scheduler.step()

        if config["impute_every"] > 0 and epoch % config["impute_every"] == 0:
            update_missing_from_recon(
                model,
                fold_data,
                edge_index,
                edge_attr,
                non_aki_mask_f,
                config["batch_size"],
            )

        if epoch % config["eval_every"] == 0 or epoch == 1:
            val_prob = score_indices(
                model,
                fold_data.x_filled,
                fold_data.input_missing,
                edge_index,
                edge_attr,
                val_idx,
                config["batch_size"],
            )
            y_val = fold_data.y[torch.as_tensor(val_idx, device=device)].cpu().numpy()
            val_auc = safe_auroc(y_val, val_prob)
            mean_loss = float(np.mean([d["loss"] for d in epoch_losses]))
            mean_rec = float(np.mean([d["rec"] for d in epoch_losses]))
            mean_aki = float(np.mean([d["aki"] for d in epoch_losses]))
            history.append({"epoch": epoch, "val_auroc": val_auc, "loss": mean_loss})

            if config.get("wandb") and HAS_WANDB:
                wandb.log(
                    {
                        f"fold{fold}/loss": mean_loss,
                        f"fold{fold}/loss_rec": mean_rec,
                        f"fold{fold}/loss_aki": mean_aki,
                        f"fold{fold}/val_auroc": val_auc,
                        f"fold{fold}/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            if np.isfinite(val_auc) and val_auc > best_val:
                best_val, best_epoch = val_auc, epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = config["patience"]
            else:
                patience_left -= 1

            if config.get("verbose"):
                print(
                    f"    ep={epoch:04d} loss={mean_loss:.4f} val={val_auc:.4f} "
                    f"best={best_val:.4f} pat={patience_left}"
                )
            if patience_left <= 0:
                break

    # ── Test ──────────────────────────────────────────────────
    model.load_state_dict(best_state)
    test_prob = score_indices(
        model,
        fold_data.x_filled,
        fold_data.input_missing,
        edge_index,
        edge_attr,
        test_idx,
        config["batch_size"],
    )
    y_test = fold_data.y[torch.as_tensor(test_idx, device=device)].cpu().numpy()
    test_auc = safe_auroc(y_test, test_prob)
    test_auprc = safe_auprc(y_test, test_prob)

    # Same-fold baselines (exact same preprocessing)
    non_aki = [i for i in range(num_features) if i != aki_idx]
    X_bl = fold_data.x_filled[:, non_aki].cpu().numpy()
    y_np = fold_data.y.cpu().numpy()

    # LogReg
    lr = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=config["logreg_C"]
    )
    lr.fit(X_bl[train_idx], y_np[train_idx])
    lr_prob = lr.predict_proba(X_bl[test_idx])[:, 1]
    lr_auc = safe_auroc(y_np[test_idx], lr_prob)
    lr_auprc = safe_auprc(y_np[test_idx], lr_prob)

    # XGBoost
    xgb_auc, xgb_auprc = float("nan"), float("nan")
    try:
        from xgboost import XGBClassifier

        n_pos_np = y_np[train_idx].sum()
        n_neg_np = len(train_idx) - n_pos_np
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=max(1, n_neg_np / max(1, n_pos_np)),
            eval_metric="logloss",
            verbosity=0,
            random_state=config["seed"],
        )
        xgb.fit(X_bl[train_idx], y_np[train_idx])
        xgb_prob = xgb.predict_proba(X_bl[test_idx])[:, 1]
        xgb_auc = safe_auroc(y_np[test_idx], xgb_prob)
        xgb_auprc = safe_auprc(y_np[test_idx], xgb_prob)
    except ImportError:
        pass

    return {
        "fold": fold,
        "auroc": test_auc,
        "auprc": test_auprc,
        "best_val_auroc": float(best_val),
        "best_epoch": int(best_epoch),
        "logreg_auroc": lr_auc,
        "logreg_auprc": lr_auprc,
        "xgb_auroc": xgb_auc,
        "xgb_auprc": xgb_auprc,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "history": history,
    }


# ──────────────────────────────────────────────────────────────
#  Main CV loop
# ──────────────────────────────────────────────────────────────


def run_cv(config):
    set_seed(config["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.get("cpu") else "cpu"
    )
    data_dir = config["data_dir"]

    X_fn = torch.load(
        os.path.join(data_dir, "feature_matrix.pt"), weights_only=True
    ).float()
    binary_mask_f = torch.load(
        os.path.join(data_dir, "binary_mask.pt"), weights_only=True
    ).bool()
    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    aki_idx = feature_names.index("aki_event")

    miss_path = os.path.join(data_dir, "missing_mask.pt")
    missing_fn = (
        torch.load(miss_path, weights_only=True).bool()
        if os.path.exists(miss_path)
        else torch.zeros_like(X_fn, dtype=torch.bool)
    )

    meta = pd.read_csv(os.path.join(data_dir, "cohort_meta.csv"))
    y_all = X_fn[aki_idx].numpy()
    surv_days = meta["surv_days"].values
    eligible = (y_all == 1) | (surv_days >= config["landmark_days"])
    eligible_idx = np.where(eligible)[0]

    X_fn = X_fn[:, eligible_idx]
    missing_fn = missing_fn[:, eligible_idx]
    y_all = y_all[eligible_idx]

    # Patient × feature layout
    X_nf = X_fn.T.contiguous()
    missing_nf = missing_fn.T.contiguous()

    recon_mode = "transductive" if not config["strict_recon"] else "strict"
    aki_mode = "recon" if config["use_recon_aki"] else "separate_head"

    print("=" * 72)
    print("Feature-graph AE:  L = λ·L_rec + L_AKI")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Data: N={X_nf.shape[0]}, F={X_nf.shape[1]}, AKI rate={y_all.mean():.3f}")
    print(
        f"Landmark: {eligible.sum()}/{len(eligible)} eligible ({config['landmark_days']}d)"
    )
    print(
        f"Graph: {config['edge_method']} k={config['k']} edge_wt={config.get('use_edge_weights', False)} "
        f"conv={config['conv_type']} L={config['layers']} self_loops={config['add_self_loops']}"
    )
    print(
        f"Loss: λ_rec={config['lambda_rec']} mask={config['mask_rate']} pseudo={config['pseudo_weight']}"
    )
    print(
        f"Mode: recon={recon_mode} aki={aki_mode} label_input={config['use_label_input']}"
    )

    # ── W&B ───────────────────────────────────────────────────
    if config.get("wandb") and HAS_WANDB:
        run_name = (
            config.get("run_name")
            or f"gae_{config['edge_method']}_l{config['layers']}_lam{config['lambda_rec']:g}"
        )
        wandb.init(
            project=config.get("wandb_project", "dualr-graph"),
            name=run_name,
            config=config,
            reinit=True,
        )

    outer = StratifiedKFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"]
    )
    fold_metrics = []

    for fold, (trainval_idx, test_idx) in enumerate(
        outer.split(np.zeros(len(y_all)), y_all), start=1
    ):
        inner = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=config["seed"] + fold
        )
        inner_train_rel, val_rel = next(
            inner.split(np.zeros(len(trainval_idx)), y_all[trainval_idx])
        )
        train_idx = trainval_idx[inner_train_rel]
        val_idx = trainval_idx[val_rel]
        print(
            f"\nFold {fold}/{config['cv_folds']}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

        metrics = train_one_fold(
            fold,
            config,
            X_nf,
            missing_nf,
            binary_mask_f,
            feature_names,
            train_idx,
            val_idx,
            test_idx,
            data_dir,
            device,
        )
        fold_metrics.append(metrics)
        print(
            f"  GNN  AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
            f"best_val={metrics['best_val_auroc']:.4f}  ep={metrics['best_epoch']}"
        )
        print(
            f"  LR   AUROC={metrics['logreg_auroc']:.4f}  AUPRC={metrics['logreg_auprc']:.4f}"
        )
        if np.isfinite(metrics.get("xgb_auroc", float("nan"))):
            print(
                f"  XGB  AUROC={metrics['xgb_auroc']:.4f}  AUPRC={metrics['xgb_auprc']:.4f}"
            )

    aurocs = np.array([m["auroc"] for m in fold_metrics], dtype=float)
    auprcs = np.array([m["auprc"] for m in fold_metrics], dtype=float)
    lr_aurocs = np.array([m["logreg_auroc"] for m in fold_metrics], dtype=float)
    xgb_aurocs = np.array(
        [m.get("xgb_auroc", float("nan")) for m in fold_metrics], dtype=float
    )

    summary = {
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "auprc_mean": float(np.nanmean(auprcs)),
        "auprc_std": float(np.nanstd(auprcs)),
        "logreg_auroc_mean": float(np.nanmean(lr_aurocs)),
        "logreg_auroc_std": float(np.nanstd(lr_aurocs)),
        "xgb_auroc_mean": float(np.nanmean(xgb_aurocs)),
        "xgb_auroc_std": float(np.nanstd(xgb_aurocs)),
        "delta_vs_logreg": float(np.nanmean(aurocs - lr_aurocs)),
        "delta_vs_xgb": float(np.nanmean(aurocs - xgb_aurocs)),
        "folds": fold_metrics,
        "config": config,
    }

    print(f"\n{'='*72}")
    print(f"GNN  AUROC: {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")
    print(f"GNN  AUPRC: {summary['auprc_mean']:.4f} ± {summary['auprc_std']:.4f}")
    print(
        f"LR   AUROC: {summary['logreg_auroc_mean']:.4f} ± {summary['logreg_auroc_std']:.4f}"
    )
    print(
        f"XGB  AUROC: {summary['xgb_auroc_mean']:.4f} ± {summary['xgb_auroc_std']:.4f}"
    )
    print(f"Δ vs LR:    {summary['delta_vs_logreg']:+.4f}")
    print(f"Δ vs XGB:   {summary['delta_vs_xgb']:+.4f}")
    print(f"{'='*72}")

    if config.get("wandb") and HAS_WANDB:
        wandb.summary.update(
            {
                "auroc_mean": summary["auroc_mean"],
                "auroc_std": summary["auroc_std"],
                "auprc_mean": summary["auprc_mean"],
                "auprc_std": summary["auprc_std"],
                "logreg_auroc_mean": summary["logreg_auroc_mean"],
                "xgb_auroc_mean": summary["xgb_auroc_mean"],
                "delta_vs_logreg": summary["delta_vs_logreg"],
                "delta_vs_xgb": summary["delta_vs_xgb"],
            }
        )
        cols = ["fold", "auroc", "auprc", "logreg_auroc", "xgb_auroc", "best_epoch"]
        table = wandb.Table(columns=cols)
        for m in fold_metrics:
            table.add_data(
                m["fold"],
                m["auroc"],
                m["auprc"],
                m["logreg_auroc"],
                m.get("xgb_auroc", float("nan")),
                m["best_epoch"],
            )
        wandb.log({"fold_results": table})
        wandb.finish()

    return summary


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Feature-graph AE for ICI→AKI")

    # Data
    p.add_argument("--data_dir", default="data")
    p.add_argument("--run_name", default=None)
    p.add_argument("--edge_method", default="llm", choices=["llm", "spearman", "mi"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--use_edge_weights", action="store_true")
    p.add_argument("--cv_cutoff", type=float, default=None)

    # Architecture
    p.add_argument(
        "--conv_type", default="gatv2", choices=["gcn", "gat", "gatv2", "transformer"]
    )
    p.add_argument("--layers", type=int, default=3, choices=[2, 3])
    p.add_argument("--node_emb_dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--add_self_loops", action="store_true", default=True)
    p.add_argument("--no_self_loops", dest="add_self_loops", action="store_false")

    # AKI prediction mode
    p.add_argument(
        "--separate_aki_head",
        dest="use_recon_aki",
        action="store_false",
        default=True,
        help="Ablation: use separate MLP head instead of AKI node reconstruction",
    )

    # Loss
    p.add_argument("--lambda_rec", type=float, default=0.5)
    p.add_argument("--mask_rate", type=float, default=0.20)
    p.add_argument("--pseudo_weight", type=float, default=0.05)

    # Semi-supervised defaults (whiteboard design)
    p.add_argument(
        "--strict_recon",
        action="store_true",
        default=False,
        help="Ablation: only train patients in L_rec (default: transductive)",
    )
    p.add_argument(
        "--no_label_input",
        dest="use_label_input",
        action="store_false",
        default=True,
        help="Ablation: do not feed train AKI labels as input (default: labels visible)",
    )

    # Training
    p.add_argument("--impute_strategy", default="mean", choices=["mean", "median"])
    p.add_argument("--impute_every", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--max_pos_weight", type=float, default=10.0)
    p.add_argument("--logreg_C", type=float, default=1.0)

    # Infra
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--landmark_days", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument("--wandb_project", default="dualr-graph")
    return p.parse_args()


def main():
    args = parse_args()
    config = vars(args)
    summary = run_cv(config)
    os.makedirs("results", exist_ok=True)
    name = (
        args.run_name
        or f"gae_{args.edge_method}_k{args.k}_l{args.layers}_lam{args.lambda_rec:g}"
    )
    out = os.path.join("results", f"{name}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
