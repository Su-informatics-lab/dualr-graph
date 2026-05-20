#!/usr/bin/env python3
"""
train_v3.py — Training script for Su's feature-graph autoencoder v3.

Key differences from train_v2.py:
  ① Feature value reconstruction loss (not association matrix reconstruction).
  ② AKI prediction comes from reconstruction, not a separate MLP head.
  ③ DirectedWeightedConv preserves P(v_i|v_j) edge weights and direction.
  ④ Association loss is optional auxiliary regularization only.
  ⑤ Simplified: no FLAG, no PCGrad, no FGSAM, no SWA — those proved marginal.
     Focus the sweep on the new loss structure.

Usage:
  python train_v3.py --edge_method llm --use_edge_weights \\
      --lambda_rec 1.0 --lambda_aki 1.0 --lambda_assoc 0.5

Sweep:
  python train_v3.py --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from graph import build_mi, build_spearman, knn_sparsify, load_llm_graph
from models_v3 import FeatureGraphAutoencoderV3, build_model_v3, total_loss

# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auroc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y, p) -> float:
    return (
        float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    )


# ══════════════════════════════════════════════════════════════
# Preprocessing (same logic as v2)
# ══════════════════════════════════════════════════════════════


def preprocess_fold(X_nf, missing_nf, binary_mask_f, train_idx):
    """Train-only z-score for continuous + mode imputation for binary."""
    X = X_nf.float().numpy().copy()
    missing = missing_nf.bool().numpy()
    binary = binary_mask_f.bool().numpy()
    observed = ~missing

    x_scaled = np.zeros_like(X, dtype=np.float32)
    obs_mask = observed.astype(np.float32)

    for j in range(X.shape[1]):
        train_obs = train_idx[observed[train_idx, j]]
        vals = X[train_obs, j].astype(np.float64)

        if binary[j]:
            fill = float(np.round(vals.mean())) if len(vals) > 0 else 0.0
            col = X[:, j].copy()
            col[missing[:, j]] = fill
            x_scaled[:, j] = col.astype(np.float32)
        else:
            mean = float(vals.mean()) if len(vals) > 0 else 0.0
            std = float(vals.std()) if len(vals) > 0 else 1.0
            if std < 1e-6:
                std = 1.0
            fill = float(np.median(vals)) if len(vals) > 0 else 0.0
            col = X[:, j].copy()
            col[observed[:, j]] = (col[observed[:, j]] - mean) / std
            col[missing[:, j]] = (fill - mean) / std
            x_scaled[:, j] = col.astype(np.float32)

    return x_scaled, obs_mask


def build_association_target(x_train_scaled: np.ndarray):
    """Train-set |Pearson corr| for optional auxiliary loss."""
    corr = np.corrcoef(x_train_scaled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    assoc = np.abs(corr).astype(np.float32)
    np.fill_diagonal(assoc, 1.0)
    return torch.from_numpy(assoc)


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════


class FeatureGraphDataset(Dataset):
    """Each patient is one graph; nodes are features."""

    def __init__(self, x_scaled, obs_mask, y, edge_index, edge_weight=None):
        super().__init__()
        self.x_scaled = torch.from_numpy(x_scaled).float()
        self.obs_mask = torch.from_numpy(obs_mask).float()
        self.y = torch.from_numpy(y).long()
        self.edge_index = edge_index.long()
        self.edge_weight = edge_weight
        self.n_features = x_scaled.shape[1]
        self.feature_id = torch.arange(self.n_features, dtype=torch.long)

    def len(self):
        return self.x_scaled.shape[0]

    def get(self, idx):
        values = self.x_scaled[idx].view(-1, 1)
        mask = self.obs_mask[idx].view(-1, 1)
        data = Data(
            x=torch.cat([values, mask], dim=1),  # [F, 2]
            edge_index=self.edge_index,
            y=self.y[idx].view(1),
            feature_id=self.feature_id,
        )
        if self.edge_weight is not None:
            data.edge_weight = self.edge_weight
        return data


def make_loader(dataset, indices, batch_size, shuffle):
    return DataLoader(
        dataset.index_select(indices.tolist()), batch_size=batch_size, shuffle=shuffle
    )


# ══════════════════════════════════════════════════════════════
# Graph construction
# ══════════════════════════════════════════════════════════════


def make_graph_for_fold(config, x_train_scaled, data_dir):
    method = config["edge_method"]
    if method in ("spearman", "mi"):
        adj = (
            build_spearman(x_train_scaled)
            if method == "spearman"
            else build_mi(x_train_scaled)
        )
        edge_index, edge_weight = knn_sparsify(adj, k=config["k"], directed=False)
    elif method == "llm":
        edge_index, edge_weight = load_llm_graph(
            data_dir=data_dir,
            k=config["k"],
            cv_cutoff=config.get("cv_cutoff"),
        )
    else:
        raise ValueError(f"Unknown edge_method={method!r}")
    return edge_index.long(), edge_weight.float()


# ══════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════


def run_epoch(
    model: FeatureGraphAutoencoderV3,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: dict,
    device: torch.device,
    train: bool,
    binary_mask_f: torch.Tensor,
    assoc_target: torch.Tensor | None,
):
    """One epoch of training or evaluation.

    x_true_all / obs_mask_all: [N, F] tensors for the FULL dataset.
    We index into them per-batch using batch.y indices... actually we need
    the original values. Let me store them in the dataset.

    Actually — the batch already has batch.x which contains [value, obs_mask].
    But AKI is masked to [0,0] in batch.x. We need the ORIGINAL x (before AKI
    masking) as reconstruction target and the original obs_mask.

    Strategy: pass x_true and obs_mask as separate tensors indexed by batch.
    """
    model.train(train)

    totals = {"loss": 0.0, "rec_loss": 0.0, "aki_loss": 0.0, "assoc_loss": 0.0}
    y_true, y_prob = [], []

    for batch in loader:
        batch = batch.to(device)
        B = batch.num_graphs
        F_dim = model.n_features

        if train:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass — edge_weight is inside batch (PyG batched it).
        out = model(batch)

        # Reconstruct x_true and obs_mask for this batch from batch.x.
        # batch.x is [B*F, 2], reshaped to [B, F, 2].
        batch_x_2d = batch.x.view(B, F_dim, 2)
        # x_true for non-AKI features comes from the batch input (already correct).
        # For AKI, the input was masked to [0,0], but we have y as the true label.
        x_true_batch = batch_x_2d[:, :, 0]  # [B, F] — values
        obs_mask_batch = batch_x_2d[:, :, 1]  # [B, F] — obs flags

        aki_true = batch.y.view(-1)  # [B]

        parts = total_loss(
            out=out,
            x_true=x_true_batch,
            obs_mask=obs_mask_batch,
            binary_mask=binary_mask_f,
            aki_true=aki_true,
            aki_idx=model.aki_idx,
            assoc_target=assoc_target,
            lambda_rec=config["lambda_rec"],
            lambda_aki=config["lambda_aki"],
            lambda_assoc=config["lambda_assoc"],
        )

        if train and torch.isfinite(parts["loss"]):
            parts["loss"].backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

        n = B
        for k in totals:
            totals[k] += parts[k].detach().item() * n

        y_prob.extend(out.aki_prob.detach().cpu().numpy().tolist())
        y_true.extend(aki_true.detach().cpu().numpy().tolist())

    n_total = max(1, len(y_true))
    metrics = {k: v / n_total for k, v in totals.items()}
    metrics["accuracy"] = accuracy_score(y_true, (np.array(y_prob) > 0.5).astype(int))
    metrics["auroc"] = safe_auroc(y_true, y_prob)
    metrics["auprc"] = safe_auprc(y_true, y_prob)
    return metrics


# ══════════════════════════════════════════════════════════════
# Baselines
# ══════════════════════════════════════════════════════════════


def run_baselines(x_scaled, y, train_idx, val_idx, test_idx, aki_idx, config):
    non_aki = [i for i in range(x_scaled.shape[1]) if i != aki_idx]
    X_bl = x_scaled[:, non_aki]

    best_lr_auc, best_lr = -1.0, None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        lr = LogisticRegression(
            max_iter=5000, class_weight="balanced", C=C, solver="lbfgs"
        )
        lr.fit(X_bl[train_idx], y[train_idx])
        val_prob = lr.predict_proba(X_bl[val_idx])[:, 1]
        val_auc = safe_auroc(y[val_idx], val_prob)
        if np.isfinite(val_auc) and val_auc > best_lr_auc:
            best_lr_auc, best_lr = val_auc, lr

    if best_lr is None:
        best_lr = LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0, solver="lbfgs"
        )
        best_lr.fit(X_bl[train_idx], y[train_idx])

    lr_prob = best_lr.predict_proba(X_bl[test_idx])[:, 1]
    lr_auc = safe_auroc(y[test_idx], lr_prob)
    lr_auprc = safe_auprc(y[test_idx], lr_prob)

    xgb_auc, xgb_auprc = float("nan"), float("nan")
    try:
        from xgboost import XGBClassifier

        n_pos = int(y[train_idx].sum())
        n_neg = int(len(train_idx) - n_pos)
        spw = max(1.0, n_neg / max(1, n_pos))

        best_xgb_val, best_xgb = -1.0, None
        for md in [3, 4, 6]:
            for lr_rate in [0.01, 0.05, 0.1]:
                xgb = XGBClassifier(
                    n_estimators=1000,
                    max_depth=md,
                    learning_rate=lr_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=spw,
                    min_child_weight=5,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=config["seed"],
                    early_stopping_rounds=30,
                )
                xgb.fit(
                    X_bl[train_idx],
                    y[train_idx],
                    eval_set=[(X_bl[val_idx], y[val_idx])],
                    verbose=False,
                )
                val_prob = xgb.predict_proba(X_bl[val_idx])[:, 1]
                val_auc = safe_auroc(y[val_idx], val_prob)
                if np.isfinite(val_auc) and val_auc > best_xgb_val:
                    best_xgb_val, best_xgb = val_auc, xgb

        if best_xgb is not None:
            xgb_prob = best_xgb.predict_proba(X_bl[test_idx])[:, 1]
            xgb_auc = safe_auroc(y[test_idx], xgb_prob)
            xgb_auprc = safe_auprc(y[test_idx], xgb_prob)
    except Exception as e:
        if config.get("verbose"):
            print(f"  XGB skipped/failed: {e}")

    return lr_auc, lr_auprc, xgb_auc, xgb_auprc


# ══════════════════════════════════════════════════════════════
# Single fold
# ══════════════════════════════════════════════════════════════


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

    x_scaled, obs_mask = preprocess_fold(X_nf, missing_nf, binary_mask_f, train_idx)
    y = X_nf[:, aki_idx].float().numpy().astype(np.int64)

    # Build graph topology (LLM directed edges).
    edge_index, edge_weight = make_graph_for_fold(config, x_scaled[train_idx], data_dir)

    # Build association target for optional auxiliary loss.
    assoc_target = None
    if config["lambda_assoc"] > 0:
        assoc_target = build_association_target(x_scaled[train_idx]).to(device)

    # Mask AKI AFTER building association target.
    x_scaled[:, aki_idx] = 0.0
    obs_mask[:, aki_idx] = 0.0

    # Edge weight for directed conv — stored in each Data, PyG batches it.
    ew_for_dataset = edge_weight if config["use_edge_weights"] else None

    dataset = FeatureGraphDataset(
        x_scaled, obs_mask, y, edge_index, edge_weight=ew_for_dataset
    )
    train_loader = make_loader(dataset, train_idx, config["batch_size"], shuffle=True)
    val_loader = make_loader(dataset, val_idx, config["batch_size"], shuffle=False)
    test_loader = make_loader(dataset, test_idx, config["batch_size"], shuffle=False)

    binary_mask_device = binary_mask_f.to(device)

    model = build_model_v3(
        n_features=num_features,
        aki_idx=aki_idx,
        hidden_dim=config["hidden"],
        embedding_dim=config["latent"],
        n_layers=config["layers"],
        dropout=config["dropout"],
        use_assoc_aux=(config["lambda_assoc"] > 0),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    best_val_auc, best_epoch, best_state = -1.0, -1, None
    patience_left = config["patience"]

    for epoch in range(1, config["epochs"] + 1):
        train_m = run_epoch(
            model,
            train_loader,
            optimizer,
            config,
            device,
            train=True,
            binary_mask_f=binary_mask_device,
            assoc_target=assoc_target,
        )
        scheduler.step()

        if epoch == 1 or epoch % config["eval_every"] == 0:
            val_m = run_epoch(
                model,
                val_loader,
                None,
                config,
                device,
                train=False,
                binary_mask_f=binary_mask_device,
                assoc_target=assoc_target,
            )

            if config.get("wandb") and HAS_WANDB:
                step = (fold - 1) * config["epochs"] + epoch
                wandb.log(
                    {
                        f"fold{fold}/train_loss": train_m["loss"],
                        f"fold{fold}/train_rec": train_m["rec_loss"],
                        f"fold{fold}/train_aki": train_m["aki_loss"],
                        f"fold{fold}/train_auroc": train_m["auroc"],
                        f"fold{fold}/val_loss": val_m["loss"],
                        f"fold{fold}/val_auroc": val_m["auroc"],
                        f"fold{fold}/val_auprc": val_m["auprc"],
                    },
                    step=step,
                )

            if np.isfinite(val_m["auroc"]) and val_m["auroc"] > best_val_auc:
                best_val_auc = val_m["auroc"]
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_left = config["patience"]
            else:
                patience_left -= 1

            if config.get("verbose"):
                print(
                    f"    ep={epoch:03d} val_auc={val_m['auroc']:.4f} "
                    f"best={best_val_auc:.4f} rec={val_m['rec_loss']:.4f} "
                    f"aki={val_m['aki_loss']:.4f} assoc={val_m['assoc_loss']:.4f} "
                    f"pat={patience_left}"
                )

            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_m = run_epoch(
        model,
        test_loader,
        None,
        config,
        device,
        train=False,
        binary_mask_f=binary_mask_device,
        assoc_target=assoc_target,
    )

    lr_auc, lr_auprc, xgb_auc, xgb_auprc = run_baselines(
        x_scaled,
        y,
        train_idx,
        val_idx,
        test_idx,
        aki_idx,
        config,
    )

    return {
        "fold": fold,
        "auroc": test_m["auroc"],
        "auprc": test_m["auprc"],
        "accuracy": test_m["accuracy"],
        "rec_loss": test_m["rec_loss"],
        "aki_loss": test_m["aki_loss"],
        "assoc_loss": test_m["assoc_loss"],
        "best_val_auroc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "logreg_auroc": lr_auc,
        "logreg_auprc": lr_auprc,
        "xgb_auroc": xgb_auc,
        "xgb_auprc": xgb_auprc,
    }


# ══════════════════════════════════════════════════════════════
# CV loop
# ══════════════════════════════════════════════════════════════


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
    if os.path.exists(miss_path):
        missing_fn = torch.load(miss_path, weights_only=True).bool()
    else:
        missing_fn = torch.zeros_like(X_fn, dtype=torch.bool)

    meta = pd.read_csv(os.path.join(data_dir, "cohort_meta.csv"))
    y_all = X_fn[aki_idx].cpu().numpy().astype(np.int64)
    eligible = (y_all == 1) | (meta["surv_days"].values >= config["landmark_days"])
    eligible_idx = np.where(eligible)[0]

    X_fn = X_fn[:, eligible_idx]
    missing_fn = missing_fn[:, eligible_idx]
    y_all = y_all[eligible_idx]
    X_nf = X_fn.T.contiguous()
    missing_nf = missing_fn.T.contiguous()

    print("=" * 72)
    print(f"v3 | DirConv + FeatureValueDecoder + AKI-from-recon")
    print(
        f"λ_rec={config['lambda_rec']} λ_aki={config['lambda_aki']} "
        f"λ_assoc={config['lambda_assoc']} dropout={config['dropout']}"
    )
    print(
        f"topology={config['edge_method']} k={config['k']} "
        f"edge_weights={config['use_edge_weights']}"
    )
    print(f"N={X_nf.shape[0]} F={X_nf.shape[1]} AKI={y_all.mean():.3f} device={device}")
    print("=" * 72)

    if config.get("wandb") and HAS_WANDB:
        run_name = (
            config.get("run_name")
            or f"v10_{config['lambda_rec']}_{config['lambda_aki']}_{config['lambda_assoc']}"
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
        outer.split(np.zeros(len(y_all)), y_all), 1
    ):
        inner = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=config["seed"] + fold
        )
        train_rel, val_rel = next(
            inner.split(np.zeros(len(trainval_idx)), y_all[trainval_idx])
        )
        train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]

        print(
            f"\nFold {fold}/{config['cv_folds']}: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
        t0 = time.time()
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
        metrics["time_s"] = time.time() - t0
        fold_metrics.append(metrics)

        print(f"  GNN  AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}")
        print(f"  LR   AUROC={metrics['logreg_auroc']:.4f}")
        if np.isfinite(metrics.get("xgb_auroc", float("nan"))):
            print(f"  XGB  AUROC={metrics['xgb_auroc']:.4f}")

    aurocs = np.array([m["auroc"] for m in fold_metrics])
    lr_aurocs = np.array([m["logreg_auroc"] for m in fold_metrics])
    xgb_aurocs = np.array([m.get("xgb_auroc", float("nan")) for m in fold_metrics])

    summary = {
        "strategy": config.get("run_name") or "v10",
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "auprc_mean": float(np.nanmean([m["auprc"] for m in fold_metrics])),
        "auprc_std": float(np.nanstd([m["auprc"] for m in fold_metrics])),
        "logreg_auroc_mean": float(np.nanmean(lr_aurocs)),
        "delta_vs_logreg": float(np.nanmean(aurocs - lr_aurocs)),
        "folds": fold_metrics,
        "config": config,
    }
    if np.any(np.isfinite(xgb_aurocs)):
        summary["xgb_auroc_mean"] = float(np.nanmean(xgb_aurocs))
        summary["delta_vs_xgb"] = float(np.nanmean(aurocs - xgb_aurocs))

    print(f"\n{'='*72}")
    print(f"GNN  AUROC: {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")
    print(f"LR   AUROC: {summary['logreg_auroc_mean']:.4f}")
    if "xgb_auroc_mean" in summary:
        print(f"XGB  AUROC: {summary['xgb_auroc_mean']:.4f}")
    print(f"Δ vs LR:    {summary['delta_vs_logreg']:+.4f}")
    print(f"{'='*72}")

    if config.get("wandb") and HAS_WANDB:
        wandb.summary.update(
            {k: v for k, v in summary.items() if k not in ("folds", "config")}
        )
        wandb.finish()

    return summary


# ══════════════════════════════════════════════════════════════
# Sweep configs
# ══════════════════════════════════════════════════════════════


def generate_sweep_configs(base_config: dict) -> list[dict]:
    """Generate sweep grid for v3."""
    configs = []

    # The sweep explores the new loss structure.
    # Architecture: use v5 winners (hidden=64, latent=16, layers=2, lr=1e-3).
    # Topology: LLM edges with weights (proven best).
    sweep_grid = {
        "lambda_rec": [0.5, 1.0, 2.0],
        "lambda_aki": [1.0, 2.0],
        "lambda_assoc": [0.0, 0.5, 1.0],  # 0 = no auxiliary
        "dropout": [0.0, 0.1],
    }

    from itertools import product

    keys = list(sweep_grid.keys())
    for vals in product(*[sweep_grid[k] for k in keys]):
        c = base_config.copy()
        for k, v in zip(keys, vals):
            c[k] = v
        name = f"v10_rec{c['lambda_rec']}_aki{c['lambda_aki']}_assoc{c['lambda_assoc']}_do{c['dropout']}"
        c["run_name"] = name
        configs.append(c)

    return configs


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="Feature-graph AE v3 (Su's design)")

    # Data
    p.add_argument("--data_dir", default="data")
    p.add_argument("--landmark_days", type=int, default=180)

    # Topology
    p.add_argument("--edge_method", default="llm", choices=["llm", "spearman", "mi"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--use_edge_weights", action="store_true")
    p.add_argument("--cv_cutoff", type=float, default=None)

    # Model
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)

    # Loss weights
    p.add_argument("--lambda_rec", type=float, default=1.0)
    p.add_argument("--lambda_aki", type=float, default=1.0)
    p.add_argument("--lambda_assoc", type=float, default=0.5)

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=5.0)

    # CV / infra
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="dualr-graph")
    p.add_argument("--run_name", default=None)

    # Sweep mode
    p.add_argument("--sweep", action="store_true", help="Run full sweep grid")

    return p.parse_args()


def main():
    args = parse_args()
    config = vars(args)

    if config.pop("sweep", False):
        # Run all sweep configs.
        base = config.copy()
        configs = generate_sweep_configs(base)
        print(f"Sweep: {len(configs)} configs")
        all_summaries = []
        for i, c in enumerate(configs, 1):
            print(f"\n{'#'*72}")
            print(f"Sweep {i}/{len(configs)}: {c['run_name']}")
            print(f"{'#'*72}")
            summary = run_cv(c)
            all_summaries.append(summary)
            # Save each result individually.
            os.makedirs("results", exist_ok=True)
            with open(os.path.join("results", f"{c['run_name']}.json"), "w") as f:
                json.dump(summary, f, indent=2)

        # Print sweep summary sorted by AUROC.
        print(f"\n{'='*72}")
        print("SWEEP SUMMARY (sorted by AUROC)")
        print(f"{'='*72}")
        ranked = sorted(all_summaries, key=lambda s: s["auroc_mean"], reverse=True)
        for s in ranked:
            print(
                f"  {s['auroc_mean']:.4f} ± {s['auroc_std']:.4f}  "
                f"Δ_LR={s['delta_vs_logreg']:+.4f}  "
                f"{s['strategy']}"
            )
    else:
        summary = run_cv(config)
        os.makedirs("results", exist_ok=True)
        name = args.run_name or "v10"
        with open(os.path.join("results", f"{name}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: results/{name}.json")


if __name__ == "__main__":
    main()
