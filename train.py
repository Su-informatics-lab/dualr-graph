#!/usr/bin/env python3
"""
train.py — Training strategies for the feature-graph autoencoder.

Strategies: joint, pretrain_finetune, embed_xgb

v3.1 fixes (GPT deep search):
  ① λ_rec sweep includes 0.017 (analytical gradient balance)
  ② Type-aware recon: BCE for binary features, MSE for continuous
  ④ Feature-type embedding in model (handled by models.py)
  ⑤ Gradient ratio monitoring: ∥λ∇L_rec∥ / ∥∇L_AKI∥ → wandb
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time

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
from models import UncertaintyWeighting, build_model

# ──────────────────────────────────────────────────────────────
#  Utilities
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
        self.x_filled, self.x_target = x_filled, x_target
        self.observed_mask, self.input_missing = observed_mask, input_missing
        self.y, self.train_prior, self.stats = y, train_prior, stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y, p):
    return (
        float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    )


def iter_batches(indices, batch_size, shuffle=True):
    idx = np.random.permutation(indices) if shuffle else np.asarray(indices)
    for s in range(0, len(idx), batch_size):
        yield idx[s : s + batch_size]


def _mode_or_mean(v):
    if v.numel() == 0:
        return 0.0
    vals, cts = torch.unique(v.float(), return_counts=True)
    return float(vals[cts.argmax()].item())


# ──────────────────────────────────────────────────────────────
#  Preprocessing
# ──────────────────────────────────────────────────────────────


def preprocess_fold(
    X_nf,
    missing_nf,
    binary_mask_f,
    train_idx,
    aki_idx,
    impute_strategy="mean",
    use_label_input=True,
):
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
    stats = {}
    for j in range(F):
        if j == aki_idx:
            continue
        obs_train = train_t[~missing_nf[train_t, j]]
        vals_train = X_nf[obs_train, j].float()
        if binary_mask_f[j]:
            fill = _mode_or_mean(vals_train)
            x_filled[missing_nf[:, j], j] = fill
            x_target[missing_nf[:, j], j] = fill
        else:
            mean = float(vals_train.mean()) if vals_train.numel() else 0.0
            std = float(vals_train.std(unbiased=False)) if vals_train.numel() else 1.0
            if std < 1e-6:
                std = 1.0
            fill_raw = (
                float(vals_train.median())
                if impute_strategy == "median" and vals_train.numel()
                else mean
            )
            obs_all = ~missing_nf[:, j]
            x_filled[obs_all, j] = (X_nf[obs_all, j] - mean) / std
            x_target[obs_all, j] = x_filled[obs_all, j]
            fill_std = (fill_raw - mean) / std
            x_filled[missing_nf[:, j], j] = fill_std
            x_target[missing_nf[:, j], j] = fill_std

    if use_label_input:
        x_filled[:, aki_idx] = prior
        x_filled[train_t, aki_idx] = y[train_t]
        input_missing[:, aki_idx] = True
        input_missing[train_t, aki_idx] = False
    else:
        x_filled[:, aki_idx] = prior
        input_missing[:, aki_idx] = True
    x_target[:, aki_idx] = prior
    observed[:, aki_idx] = False
    return FoldData(x_filled, x_target, observed, input_missing, y, prior, stats)


def make_graph_for_fold(config, fold_data, train_idx, aki_idx, data_dir):
    method = config["edge_method"]
    if method in ("spearman", "mi"):
        x_graph = fold_data.x_filled.clone()
        x_graph[:, aki_idx] = fold_data.train_prior
        x_graph[train_idx, aki_idx] = fold_data.y[train_idx]
        X_train_np = x_graph[train_idx].cpu().numpy()
        adj = (
            build_spearman(X_train_np) if method == "spearman" else build_mi(X_train_np)
        )
        edge_index, edge_weight = knn_sparsify(adj, k=config["k"], directed=False)
    else:
        edge_index, edge_weight = load_llm_graph(
            data_dir=data_dir, k=config["k"], cv_cutoff=config.get("cv_cutoff")
        )
    return edge_index.long(), edge_weight.float()


def denoise_batch(
    x_filled, input_missing, observed_mask, batch_idx, non_aki_mask_f, mask_rate
):
    idx = torch.as_tensor(batch_idx, dtype=torch.long, device=x_filled.device)
    x_b = x_filled[idx].clone()
    miss_b = input_missing[idx].clone()
    if mask_rate > 0:
        eligible = observed_mask[idx] & non_aki_mask_f.unsqueeze(0)
        drop = (torch.rand_like(x_b) < mask_rate) & eligible
        x_b[drop] = 0.0
        miss_b = miss_b | drop
    return x_b, miss_b


@torch.no_grad()
def score_indices(
    model, x_filled, input_missing, edge_index, edge_attr, indices, batch_size
):
    model.eval()
    probs = []
    for bi in iter_batches(indices, batch_size, shuffle=False):
        idx = torch.as_tensor(bi, dtype=torch.long, device=x_filled.device)
        out = model(x_filled[idx], edge_index, edge_attr, input_missing[idx])
        probs.append(torch.sigmoid(out.aki_logits).cpu().numpy())
    return np.concatenate(probs)


# ──────────────────────────────────────────────────────────────
#  Fix #2: Type-aware reconstruction loss
# ──────────────────────────────────────────────────────────────


def _typed_rec_loss(
    x_hat, target, observed, non_aki, binary_mask_f, pseudo_weight, device
):
    """
    BCE for binary features, MSE for continuous features.
    Returns mean-reduced loss across all non-AKI observed cells.
    """
    non_aki_d = non_aki.to(device).unsqueeze(0)  # [1, F]
    obs = observed.to(device)  # [B, F]
    bin_mask = binary_mask_f.to(device).unsqueeze(0)  # [1, F]

    # Weight masks
    obs_non_aki = obs & non_aki_d  # [B, F]
    pseudo_mask = (~obs) & non_aki_d  # [B, F]

    # Binary features: BCE (sigmoid already applied inside bce_with_logits)
    bin_cells = obs_non_aki & bin_mask  # [B, F]
    # Continuous features: MSE
    cont_cells = obs_non_aki & (~bin_mask)  # [B, F]

    loss = torch.tensor(0.0, device=device)
    n_cells = 0.0

    if bin_cells.any():
        bce = F.binary_cross_entropy_with_logits(
            x_hat[bin_cells], target[bin_cells], reduction="sum"
        )
        loss = loss + bce
        n_cells += bin_cells.float().sum().item()

    if cont_cells.any():
        mse = (x_hat[cont_cells] - target[cont_cells]).pow(2).sum()
        loss = loss + mse
        n_cells += cont_cells.float().sum().item()

    # Pseudo-targets for missing cells (use MSE regardless)
    if pseudo_weight > 0 and pseudo_mask.any():
        pseudo_mse = (
            pseudo_weight * (x_hat[pseudo_mask] - target[pseudo_mask]).pow(2).sum()
        )
        loss = loss + pseudo_mse
        n_cells += pseudo_weight * pseudo_mask.float().sum().item()

    if n_cells > 0:
        loss = loss / n_cells

    return loss


def compute_loss(
    out,
    batch_idx,
    fold_data,
    train_indicator,
    non_aki_mask_f,
    binary_mask_f,
    lambda_rec,
    pseudo_weight,
    pos_weight,
    aki_weight=1.0,
    uw=None,
):
    """L = λ·L_rec(type-aware) + α·L_AKI"""
    device = out.x_hat.device
    idx = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
    target = fold_data.x_target[idx].to(device)
    observed = fold_data.observed_mask[idx]

    if lambda_rec > 0:
        rec_loss = _typed_rec_loss(
            out.x_hat,
            target,
            observed,
            non_aki_mask_f,
            binary_mask_f,
            pseudo_weight,
            device,
        )
    else:
        rec_loss = torch.tensor(0.0, device=device)

    ce_mask = train_indicator[idx]
    if ce_mask.any():
        y = fold_data.y[idx].to(device)
        aki_loss = F.binary_cross_entropy_with_logits(
            out.aki_logits[ce_mask], y[ce_mask], pos_weight=pos_weight
        )
    else:
        aki_loss = torch.tensor(0.0, device=device)

    if uw is not None:
        loss = uw(rec_loss, aki_loss)
    else:
        loss = lambda_rec * rec_loss + aki_weight * aki_loss

    return loss, rec_loss, aki_loss


def compute_rec_only_loss(
    out, batch_idx, fold_data, non_aki_mask_f, binary_mask_f, pseudo_weight
):
    """Reconstruction only (pretraining). Type-aware."""
    device = out.x_hat.device
    idx = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
    target = fold_data.x_target[idx].to(device)
    observed = fold_data.observed_mask[idx]
    return _typed_rec_loss(
        out.x_hat,
        target,
        observed,
        non_aki_mask_f,
        binary_mask_f,
        pseudo_weight,
        device,
    )


def run_baselines(fold_data, train_idx, test_idx, num_features, aki_idx, config):
    non_aki = [i for i in range(num_features) if i != aki_idx]
    X_bl = fold_data.x_filled[:, non_aki].cpu().numpy()
    y_np = fold_data.y.cpu().numpy()

    lr = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=config["logreg_C"]
    )
    lr.fit(X_bl[train_idx], y_np[train_idx])
    lr_prob = lr.predict_proba(X_bl[test_idx])[:, 1]
    lr_auc, lr_auprc = safe_auroc(y_np[test_idx], lr_prob), safe_auprc(
        y_np[test_idx], lr_prob
    )

    xgb_auc, xgb_auprc = float("nan"), float("nan")
    try:
        from xgboost import XGBClassifier

        n_pos = y_np[train_idx].sum()
        n_neg = len(train_idx) - n_pos
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=max(1, n_neg / max(1, n_pos)),
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
    return lr_auc, lr_auprc, xgb_auc, xgb_auprc


# ──────────────────────────────────────────────────────────────
#  Fix #5: Gradient ratio monitoring
# ──────────────────────────────────────────────────────────────


def _grad_ratio(model, rec_loss, aki_loss, lambda_rec, aki_weight):
    """Compute ∥λ∇L_rec∥ / ∥∇L_AKI∥ on shared encoder params."""
    shared = [p for p in model.encoder.parameters() if p.requires_grad]
    if not shared or not rec_loss.requires_grad or not aki_loss.requires_grad:
        return float("nan")

    g_rec = torch.autograd.grad(
        lambda_rec * rec_loss, shared, retain_graph=True, allow_unused=True
    )
    g_aki = torch.autograd.grad(
        aki_weight * aki_loss, shared, retain_graph=True, allow_unused=True
    )

    norm_rec = sum(
        (g.norm() ** 2 for g in g_rec if g is not None), torch.tensor(0.0)
    ).sqrt()
    norm_aki = sum(
        (g.norm() ** 2 for g in g_aki if g is not None), torch.tensor(0.0)
    ).sqrt()

    if norm_aki.item() < 1e-10:
        return float("inf")
    return float((norm_rec / norm_aki).item())


# ──────────────────────────────────────────────────────────────
#  Strategy: JOINT
# ──────────────────────────────────────────────────────────────


def _train_joint(
    fold,
    model,
    fold_data,
    edge_index,
    edge_attr,
    train_idx,
    val_idx,
    non_aki_mask_f,
    binary_mask_f,
    config,
    device,
):
    N = fold_data.x_filled.size(0)
    train_indicator = torch.zeros(N, dtype=torch.bool, device=device)
    train_indicator[torch.as_tensor(train_idx, device=device)] = True
    fit_idx = np.arange(N) if not config["strict_recon"] else train_idx

    y_train = fold_data.y[torch.as_tensor(train_idx, device=device)]
    pos_weight = (
        (y_train == 0).sum().float() / (y_train == 1).sum().float().clamp_min(1)
    ).clamp(max=config["max_pos_weight"])

    uw = None
    params = list(model.parameters())
    if config.get("uw"):
        uw = UncertaintyWeighting().to(device)
        params += list(uw.parameters())

    optimizer = torch.optim.AdamW(
        params, lr=config["lr"], weight_decay=config["weight_decay"]
    )
    warmup_ep = config.get("warmup_epochs", 20)

    def lr_lambda(ep):
        if ep < warmup_ep:
            return max(0.1, ep / warmup_ep)
        prog = (ep - warmup_ep) / max(1, config["epochs"] - warmup_ep)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val, best_epoch = -np.inf, -1
    best_state = copy.deepcopy(model.state_dict())
    patience_left = config["patience"]

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        if uw:
            uw.train()
        epoch_losses, epoch_ratios = [], []
        for bi in iter_batches(fit_idx, config["batch_size"], shuffle=True):
            x_b, miss_b = denoise_batch(
                fold_data.x_filled,
                fold_data.input_missing,
                fold_data.observed_mask,
                bi,
                non_aki_mask_f,
                config["mask_rate"],
            )
            out = model(x_b, edge_index, edge_attr, miss_b)
            loss, rec_l, aki_l = compute_loss(
                out,
                bi,
                fold_data,
                train_indicator,
                non_aki_mask_f,
                binary_mask_f,
                config["lambda_rec"],
                config["pseudo_weight"],
                pos_weight,
                config.get("aki_weight", 1.0),
                uw,
            )

            if torch.isfinite(loss) and loss.requires_grad:
                # Fix #5: gradient ratio (sample every 20th batch, cheap)
                if (
                    len(epoch_losses) % 20 == 0
                    and rec_l.requires_grad
                    and aki_l.requires_grad
                ):
                    try:
                        ratio = _grad_ratio(
                            model,
                            rec_l,
                            aki_l,
                            config["lambda_rec"],
                            config.get("aki_weight", 1.0),
                        )
                        epoch_ratios.append(ratio)
                    except RuntimeError:
                        pass

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
            epoch_losses.append(
                {
                    "loss": float(loss.detach()),
                    "rec": float(rec_l.detach()),
                    "aki": float(aki_l.detach()),
                }
            )
        scheduler.step()

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

            if config.get("wandb") and HAS_WANDB:
                log_d = {
                    f"fold{fold}/loss": np.mean([d["loss"] for d in epoch_losses]),
                    f"fold{fold}/loss_rec": np.mean([d["rec"] for d in epoch_losses]),
                    f"fold{fold}/loss_aki": np.mean([d["aki"] for d in epoch_losses]),
                    f"fold{fold}/val_auroc": val_auc,
                    "epoch": epoch,
                }
                if epoch_ratios:
                    log_d[f"fold{fold}/grad_ratio"] = np.mean(epoch_ratios)
                if uw:
                    log_d[f"fold{fold}/sigma_rec"] = float(
                        torch.exp(0.5 * uw.log_var_rec)
                    )
                    log_d[f"fold{fold}/sigma_aki"] = float(
                        torch.exp(0.5 * uw.log_var_aki)
                    )
                wandb.log(log_d, step=epoch)

            if np.isfinite(val_auc) and val_auc > best_val:
                best_val, best_epoch = val_auc, epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = config["patience"]
            else:
                patience_left -= 1

            if config.get("verbose"):
                gr = f" gr={np.mean(epoch_ratios):.2f}" if epoch_ratios else ""
                print(
                    f"    ep={epoch:04d} val={val_auc:.4f} best={best_val:.4f} pat={patience_left}{gr}"
                )
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return best_val, best_epoch


# ──────────────────────────────────────────────────────────────
#  Strategy: PRETRAIN → FINETUNE
# ──────────────────────────────────────────────────────────────


def _pretrain_encoder(
    model,
    fold_data,
    edge_index,
    edge_attr,
    non_aki_mask_f,
    binary_mask_f,
    config,
    device,
    fold,
):
    N = fold_data.x_filled.size(0)
    fit_idx = np.arange(N)
    pt_epochs = config.get("pretrain_epochs", 200)

    enc_dec_params = (
        list(model.encoder.parameters())
        + list(model.dec1.parameters())
        + list(model.dec2.parameters())
        + list(model.recon_head.parameters())
        + list(model.node_emb.parameters())
        + list(model.feat_type_emb.parameters())
    )
    optimizer = torch.optim.AdamW(
        enc_dec_params, lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pt_epochs)

    for epoch in range(1, pt_epochs + 1):
        model.train()
        for bi in iter_batches(fit_idx, config["batch_size"], shuffle=True):
            x_b, miss_b = denoise_batch(
                fold_data.x_filled,
                fold_data.input_missing,
                fold_data.observed_mask,
                bi,
                non_aki_mask_f,
                config["mask_rate"],
            )
            out = model(x_b, edge_index, edge_attr, miss_b)
            loss = compute_rec_only_loss(
                out,
                bi,
                fold_data,
                non_aki_mask_f,
                binary_mask_f,
                config["pseudo_weight"],
            )
            if torch.isfinite(loss) and loss.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_dec_params, config["grad_clip"])
                optimizer.step()
        scheduler.step()
        if config.get("wandb") and HAS_WANDB and (epoch % 20 == 0 or epoch == 1):
            wandb.log(
                {f"fold{fold}/pretrain_rec": float(loss.detach()), "epoch": epoch},
                step=epoch,
            )
        if config.get("verbose") and epoch % 50 == 0:
            print(f"    pretrain ep={epoch:04d} rec={float(loss.detach()):.4f}")


def _finetune_head(
    model,
    fold_data,
    edge_index,
    edge_attr,
    train_idx,
    val_idx,
    non_aki_mask_f,
    config,
    device,
    fold,
):
    N = fold_data.x_filled.size(0)
    train_indicator = torch.zeros(N, dtype=torch.bool, device=device)
    train_indicator[torch.as_tensor(train_idx, device=device)] = True

    y_train = fold_data.y[torch.as_tensor(train_idx, device=device)]
    pos_weight = (
        (y_train == 0).sum().float() / (y_train == 1).sum().float().clamp_min(1)
    ).clamp(max=config["max_pos_weight"])

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.dec1.parameters():
        p.requires_grad = False
    for p in model.dec2.parameters():
        p.requires_grad = False
    for p in model.recon_head.parameters():
        p.requires_grad = False
    for p in model.node_emb.parameters():
        p.requires_grad = False
    for p in model.feat_type_emb.parameters():
        p.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        head_params, lr=config["lr"] * 0.5, weight_decay=config["weight_decay"]
    )
    ft_epochs = config.get("finetune_epochs", 300)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)

    best_val, best_epoch = -np.inf, -1
    best_state = copy.deepcopy(model.state_dict())
    patience_left = config["patience"]

    for epoch in range(1, ft_epochs + 1):
        model.train()
        for bi in iter_batches(train_idx, config["batch_size"], shuffle=True):
            x_b = fold_data.x_filled[torch.as_tensor(bi, device=device)]
            miss_b = fold_data.input_missing[torch.as_tensor(bi, device=device)]
            out = model(x_b, edge_index, edge_attr, miss_b)
            ce_mask = train_indicator[torch.as_tensor(bi, device=device)]
            y = fold_data.y[torch.as_tensor(bi, device=device)]
            loss = F.binary_cross_entropy_with_logits(
                out.aki_logits[ce_mask], y[ce_mask], pos_weight=pos_weight
            )
            if torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        scheduler.step()

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

            if config.get("wandb") and HAS_WANDB:
                wandb.log(
                    {
                        f"fold{fold}/ft_aki_loss": float(loss.detach()),
                        f"fold{fold}/ft_val_auroc": val_auc,
                        "epoch": epoch,
                    },
                    step=epoch + config.get("pretrain_epochs", 200),
                )

            if np.isfinite(val_auc) and val_auc > best_val:
                best_val, best_epoch = val_auc, epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = config["patience"]
            else:
                patience_left -= 1
            if config.get("verbose"):
                print(
                    f"    finetune ep={epoch:04d} val={val_auc:.4f} best={best_val:.4f}"
                )
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad = True
    return best_val, best_epoch


# ──────────────────────────────────────────────────────────────
#  Strategy: EMBED → XGBoost
# ──────────────────────────────────────────────────────────────


def _embed_xgb(
    model,
    fold_data,
    edge_index,
    edge_attr,
    train_idx,
    test_idx,
    num_features,
    aki_idx,
    config,
    device,
):
    model.eval()
    N = fold_data.x_filled.size(0)
    all_embs = []
    for bi in iter_batches(np.arange(N), config["batch_size"], shuffle=False):
        idx = torch.as_tensor(bi, dtype=torch.long, device=device)
        emb = model.extract_embeddings(
            fold_data.x_filled[idx], edge_index, edge_attr, fold_data.input_missing[idx]
        )
        all_embs.append(emb)
    emb_all = np.concatenate(all_embs)

    non_aki = [i for i in range(num_features) if i != aki_idx]
    X_raw = fold_data.x_filled[:, non_aki].cpu().numpy()
    X_aug = np.concatenate([X_raw, emb_all], axis=1)
    y_np = fold_data.y.cpu().numpy()

    try:
        from xgboost import XGBClassifier

        n_pos = y_np[train_idx].sum()
        n_neg = len(train_idx) - n_pos
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=max(1, n_neg / max(1, n_pos)),
            eval_metric="logloss",
            verbosity=0,
            random_state=config["seed"],
        )
        xgb.fit(X_aug[train_idx], y_np[train_idx])
        prob = xgb.predict_proba(X_aug[test_idx])[:, 1]
    except ImportError:
        lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        lr.fit(X_aug[train_idx], y_np[train_idx])
        prob = lr.predict_proba(X_aug[test_idx])[:, 1]

    return safe_auroc(y_np[test_idx], prob), safe_auprc(y_np[test_idx], prob)


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
    binary_mask_f_d = binary_mask_f.to(device)

    fold_data = preprocess_fold(
        X_nf,
        missing_nf,
        binary_mask_f,
        train_idx,
        aki_idx,
        config["impute_strategy"],
        config["use_label_input"],
    )
    for attr in ("x_filled", "x_target", "observed_mask", "input_missing", "y"):
        setattr(fold_data, attr, getattr(fold_data, attr).to(device))

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
        aki_mode=config["aki_mode"],
        add_self_loops=config["add_self_loops"],
        drop_edge_rate=config.get("drop_edge_rate", 0.0),
    ).to(device)
    # Fix #2/#4: register binary mask for type-aware recon + type embedding
    model.register_binary_mask(binary_mask_f)

    strategy = config["strategy"]

    if strategy == "joint":
        best_val, best_epoch = _train_joint(
            fold,
            model,
            fold_data,
            edge_index,
            edge_attr,
            train_idx,
            val_idx,
            non_aki_mask_f,
            binary_mask_f_d,
            config,
            device,
        )

    elif strategy == "pretrain_finetune":
        _pretrain_encoder(
            model,
            fold_data,
            edge_index,
            edge_attr,
            non_aki_mask_f,
            binary_mask_f_d,
            config,
            device,
            fold,
        )
        best_val, best_epoch = _finetune_head(
            model,
            fold_data,
            edge_index,
            edge_attr,
            train_idx,
            val_idx,
            non_aki_mask_f,
            config,
            device,
            fold,
        )

    elif strategy == "embed_xgb":
        _pretrain_encoder(
            model,
            fold_data,
            edge_index,
            edge_attr,
            non_aki_mask_f,
            binary_mask_f_d,
            config,
            device,
            fold,
        )
        emb_auc, emb_auprc = _embed_xgb(
            model,
            fold_data,
            edge_index,
            edge_attr,
            train_idx,
            test_idx,
            num_features,
            aki_idx,
            config,
            device,
        )
        lr_auc, lr_auprc, xgb_auc, xgb_auprc = run_baselines(
            fold_data, train_idx, test_idx, num_features, aki_idx, config
        )
        return {
            "fold": fold,
            "auroc": emb_auc,
            "auprc": emb_auprc,
            "best_val_auroc": float("nan"),
            "best_epoch": -1,
            "logreg_auroc": lr_auc,
            "logreg_auprc": lr_auprc,
            "xgb_auroc": xgb_auc,
            "xgb_auprc": xgb_auprc,
        }

    # Test eval
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
    test_auc, test_auprc = safe_auroc(y_test, test_prob), safe_auprc(y_test, test_prob)

    lr_auc, lr_auprc, xgb_auc, xgb_auprc = run_baselines(
        fold_data, train_idx, test_idx, num_features, aki_idx, config
    )

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
    }


# ──────────────────────────────────────────────────────────────
#  CV loop + CLI (same structure, summary includes strategy)
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
    eligible = (y_all == 1) | (meta["surv_days"].values >= config["landmark_days"])
    eligible_idx = np.where(eligible)[0]
    X_fn = X_fn[:, eligible_idx]
    missing_fn = missing_fn[:, eligible_idx]
    y_all = y_all[eligible_idx]
    X_nf = X_fn.T.contiguous()
    missing_nf = missing_fn.T.contiguous()

    n_bin = binary_mask_f.sum().item()
    n_cont = len(binary_mask_f) - n_bin
    print("=" * 72)
    print(
        f"Feature-graph AE | strategy={config['strategy']} | aki_mode={config['aki_mode']}"
    )
    print("=" * 72)
    print(
        f"Device: {device}  N={X_nf.shape[0]}  F={X_nf.shape[1]}  AKI={y_all.mean():.3f}"
    )
    print(f"Features: {n_bin} binary + {n_cont} continuous (type-aware recon loss)")
    print(
        f"Graph: {config['edge_method']} k={config['k']} conv={config['conv_type']} "
        f"L={config['layers']} drop_edge={config.get('drop_edge_rate', 0)}"
    )
    print(
        f"Loss: λ={config['lambda_rec']} α={config.get('aki_weight', 1.0)} "
        f"uw={config.get('uw', False)} mask={config['mask_rate']}"
    )

    if config.get("wandb") and HAS_WANDB:
        run_name = config.get("run_name") or (
            f"{config['strategy']}_{config['aki_mode']}_l{config['layers']}_lam{config['lambda_rec']:g}"
        )
        wandb.init(
            project=config.get("wandb_project", "dualr-graph"),
            name=run_name,
            config=config,
            tags=[
                config["strategy"],
                config["aki_mode"],
                config["edge_method"],
                config["conv_type"],
            ],
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
        tr_rel, va_rel = next(
            inner.split(np.zeros(len(trainval_idx)), y_all[trainval_idx])
        )
        train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]
        print(
            f"\nFold {fold}/{config['cv_folds']}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
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

        has_xgb = np.isfinite(metrics.get("xgb_auroc", float("nan")))
        print(f"  GNN  AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}")
        print(f"  LR   AUROC={metrics['logreg_auroc']:.4f}")
        if has_xgb:
            print(f"  XGB  AUROC={metrics['xgb_auroc']:.4f}")

    aurocs = np.array([m["auroc"] for m in fold_metrics])
    lr_aurocs = np.array([m["logreg_auroc"] for m in fold_metrics])
    xgb_aurocs = np.array([m.get("xgb_auroc", float("nan")) for m in fold_metrics])
    has_xgb_any = np.any(np.isfinite(xgb_aurocs))

    summary = {
        "strategy": config["strategy"],
        "aki_mode": config["aki_mode"],
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "auprc_mean": float(np.nanmean([m["auprc"] for m in fold_metrics])),
        "logreg_auroc_mean": float(np.nanmean(lr_aurocs)),
        "delta_vs_logreg": float(np.nanmean(aurocs - lr_aurocs)),
        "folds": fold_metrics,
        "config": config,
    }
    if has_xgb_any:
        summary["xgb_auroc_mean"] = float(np.nanmean(xgb_aurocs))
        summary["delta_vs_xgb"] = float(np.nanmean(aurocs - xgb_aurocs))

    print(f"\n{'='*72}")
    print(f"GNN  AUROC: {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")
    print(f"LR   AUROC: {summary['logreg_auroc_mean']:.4f}")
    if has_xgb_any:
        print(f"XGB  AUROC: {summary['xgb_auroc_mean']:.4f}")
    print(f"Δ vs LR:    {summary['delta_vs_logreg']:+.4f}")
    if has_xgb_any:
        print(f"Δ vs XGB:   {summary['delta_vs_xgb']:+.4f}")
    print(f"{'='*72}")

    if config.get("wandb") and HAS_WANDB:
        wandb.summary.update(
            {k: v for k, v in summary.items() if k not in ("folds", "config")}
        )
        table = wandb.Table(
            columns=["fold", "auroc", "auprc", "logreg_auroc", "xgb_auroc"]
        )
        for m in fold_metrics:
            table.add_data(
                m["fold"],
                m["auroc"],
                m["auprc"],
                m["logreg_auroc"],
                m.get("xgb_auroc", float("nan")),
            )
        wandb.log({"fold_results": table})
        wandb.finish()

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Feature-graph AE for ICI→AKI")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--run_name", default=None)
    p.add_argument("--edge_method", default="llm", choices=["llm", "spearman", "mi"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--use_edge_weights", action="store_true")
    p.add_argument("--cv_cutoff", type=float, default=None)
    p.add_argument(
        "--strategy",
        default="joint",
        choices=["joint", "pretrain_finetune", "embed_xgb"],
    )
    p.add_argument(
        "--conv_type", default="gatv2", choices=["gcn", "gat", "gatv2", "transformer"]
    )
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--node_emb_dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--add_self_loops", action="store_true", default=True)
    p.add_argument("--no_self_loops", dest="add_self_loops", action="store_false")
    p.add_argument("--drop_edge_rate", type=float, default=0.1)
    p.add_argument(
        "--aki_mode", default="dual", choices=["recon", "recon_split", "dual", "pooled"]
    )
    p.add_argument("--lambda_rec", type=float, default=0.5)
    p.add_argument("--aki_weight", type=float, default=1.0)
    p.add_argument("--mask_rate", type=float, default=0.20)
    p.add_argument("--pseudo_weight", type=float, default=0.05)
    p.add_argument("--uw", action="store_true")
    p.add_argument("--strict_recon", action="store_true", default=False)
    p.add_argument(
        "--no_label_input", dest="use_label_input", action="store_false", default=True
    )
    p.add_argument("--impute_strategy", default="mean", choices=["mean", "median"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--pretrain_epochs", type=int, default=200)
    p.add_argument("--finetune_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--max_pos_weight", type=float, default=10.0)
    p.add_argument("--logreg_C", type=float, default=1.0)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--landmark_days", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="dualr-graph")
    return p.parse_args()


def main():
    args = parse_args()
    config = vars(args)
    summary = run_cv(config)
    os.makedirs("results", exist_ok=True)
    name = (
        args.run_name
        or f"{args.strategy}_{args.aki_mode}_{args.edge_method}_l{args.layers}_lam{args.lambda_rec:g}"
    )
    with open(os.path.join("results", f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: results/{name}.json")


if __name__ == "__main__":
    main()
