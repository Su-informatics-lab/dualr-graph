#!/usr/bin/env python3
"""
train_v2.py — Training script with audited fixes for the feature-graph AE.

Critical fixes included:
  ① Decoupled --edge_method topology from --assoc_target reconstruction target.
  ② AKI readout modes: aki_concat, aki_only, non_aki_pool, global_pool.
  ③ Static association loss on feature embeddings, selectable by --assoc_mode static.
  ④ Fixed PCGrad: cloned gradients, all shared params, optimizer.step(), optional KL/recon extras.
  ⑤ Fixed FLAG: real gradient flow through delta, perturb value channel only, never perturb AKI/missing/pretext nodes.
  ⑥ Functional masked feature reconstruction with typed losses: BCE for binary, SmoothL1 for continuous.
  ⑦ Auxiliary association-loss decay schedule.
  ⑧ Tuned LR/XGBoost baselines with validation selection.

Recommended starting command:
  python train_v2.py --edge_method llm --use_edge_weights \
      --assoc_target pearson --assoc_mode static \
      --readout aki_concat --decoder_type symmetric
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from models_v2 import association_mse, build_model, static_association_mse

# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_auroc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y, p) -> float:
    return (
        float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    )


def zero_scalar(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


# ══════════════════════════════════════════════════════════════
# FGSAM
# ══════════════════════════════════════════════════════════════


class FGSAM:
    """Small SAM/FGSAM-style wrapper around a base optimizer."""

    def __init__(
        self, params, base_optimizer, rho: float = 0.05, grad_clip: float | None = None
    ):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.grad_clip = grad_clip

    def _grad_norm(self):
        grads = [p.grad.norm(p=2) for p in self.params if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=self.params[0].device)
        return torch.norm(torch.stack(grads), p=2) + 1e-12

    @torch.no_grad()
    def _perturb(self):
        norm = self._grad_norm()
        eps_list = []
        if norm.item() == 0.0:
            return [None for _ in self.params]
        for p in self.params:
            if p.grad is None:
                eps_list.append(None)
                continue
            eps = self.rho * p.grad / norm
            p.add_(eps)
            eps_list.append(eps)
        return eps_list

    @torch.no_grad()
    def _restore(self, eps_list):
        for p, eps in zip(self.params, eps_list):
            if eps is not None:
                p.sub_(eps)

    def step(self, closure):
        eps_list = self._perturb()
        self.base_optimizer.zero_grad(set_to_none=True)
        loss = closure()
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self._restore(eps_list)
        self.base_optimizer.step()
        return loss


# ══════════════════════════════════════════════════════════════
# PCGrad
# ══════════════════════════════════════════════════════════════


def _shared_params(model):
    params = []
    params.extend(list(model.feature_embedding.parameters()))
    params.extend(list(model.input_proj.parameters()))
    if hasattr(model, "pe_proj"):
        params.extend(list(model.pe_proj.parameters()))
    params.extend(list(model.encoder.parameters()))
    if getattr(model, "use_vgae", False):
        params.extend(list(model.mu_head.parameters()))
        params.extend(list(model.logvar_head.parameters()))
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for p in params:
        if id(p) not in seen and p.requires_grad:
            unique.append(p)
            seen.add(id(p))
    return unique


def _grads_for(
    loss: torch.Tensor, params: list[torch.nn.Parameter], retain_graph: bool = True
):
    if loss is None or not torch.is_tensor(loss) or not loss.requires_grad:
        return [torch.zeros_like(p) for p in params]
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    return [
        g.detach().clone() if g is not None else torch.zeros_like(p)
        for g, p in zip(grads, params)
    ]


def pcgrad_step(
    model,
    assoc_loss_val: torch.Tensor,
    aki_loss_val: torch.Tensor,
    extra_loss_val: torch.Tensor | None,
    optimizer,
    cfg: dict[str, Any],
):
    """Project conflicting assoc-vs-AKI gradients on shared parameters, then step."""
    shared = _shared_params(model)
    optimizer.zero_grad(set_to_none=True)

    g_a = _grads_for(assoc_loss_val, shared, retain_graph=True)
    g_b = _grads_for(aki_loss_val, shared, retain_graph=True)
    g_extra = (
        _grads_for(extra_loss_val, shared, retain_graph=True)
        if extra_loss_val is not None
        else [torch.zeros_like(p) for p in shared]
    )

    flat_a = (
        torch.cat([g.flatten() for g in g_a])
        if g_a
        else torch.tensor(0.0, device=assoc_loss_val.device)
    )
    flat_b = (
        torch.cat([g.flatten() for g in g_b])
        if g_b
        else torch.tensor(0.0, device=aki_loss_val.device)
    )
    dot = (
        torch.dot(flat_a, flat_b)
        if flat_a.numel() > 1
        else torch.tensor(0.0, device=aki_loss_val.device)
    )

    if dot < 0:
        denom_b = torch.dot(flat_b, flat_b).clamp(min=1e-12)
        denom_a = torch.dot(flat_a, flat_a).clamp(min=1e-12)
        proj_ab = dot / denom_b
        proj_ba = dot / denom_a
        g_a_orig = [g.clone() for g in g_a]
        g_b_orig = [g.clone() for g in g_b]
        g_a = [ga - proj_ab * gb for ga, gb in zip(g_a_orig, g_b_orig)]
        g_b = [gb - proj_ba * ga for ga, gb in zip(g_a_orig, g_b_orig)]

    total_loss = assoc_loss_val + aki_loss_val
    if extra_loss_val is not None:
        total_loss = total_loss + extra_loss_val
    total_loss.backward()

    # Overwrite shared gradients with projected gradients plus extra-task gradients.
    for p, ga, gb, ge in zip(shared, g_a, g_b, g_extra):
        p.grad = ga + gb + ge

    if cfg["grad_clip"] > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    optimizer.step()


# ══════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════


def preprocess_fold(X_nf, missing_nf, binary_mask_f, train_idx):
    """Train-only z-score + mean/mode imputation."""
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


def build_association_target(
    assoc_target_method: str, x_train_scaled: np.ndarray, data_dir: str
):
    """Build reconstruction target independently from graph topology."""
    if assoc_target_method == "pearson":
        corr = np.corrcoef(x_train_scaled, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        assoc = np.abs(corr).astype(np.float32)
        np.fill_diagonal(assoc, 1.0)
        return torch.from_numpy(assoc), False

    if assoc_target_method == "llm":
        adj = np.load(os.path.join(data_dir, "llm_adj.npy")).astype(np.float32)
        if adj.max() > 1.0:
            adj = adj / adj.max()
        return torch.from_numpy(adj), True

    if assoc_target_method == "none":
        return None, False

    raise ValueError(f"Unknown assoc_target={assoc_target_method!r}")


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════


class FeatureGraphDataset(Dataset):
    def __init__(self, x_scaled, obs_mask, y, edge_index, edge_attr=None):
        super().__init__()
        self.x_scaled = torch.from_numpy(x_scaled).float()
        self.obs_mask = torch.from_numpy(obs_mask).float()
        self.y = torch.from_numpy(y).long()
        self.edge_index = edge_index.long()
        self.edge_attr = edge_attr
        self.n_features = x_scaled.shape[1]
        self.feature_id = torch.arange(self.n_features, dtype=torch.long)

    def len(self):
        return self.x_scaled.shape[0]

    def get(self, idx):
        values = self.x_scaled[idx].view(-1, 1)
        mask = self.obs_mask[idx].view(-1, 1)
        data = Data(
            x=torch.cat([values, mask], dim=1),
            edge_index=self.edge_index,
            y=self.y[idx].view(1),
            feature_id=self.feature_id,
        )
        if self.edge_attr is not None:
            data.edge_attr = self.edge_attr
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
# Loss helpers
# ══════════════════════════════════════════════════════════════


def _compute_assoc_loss(model, out, assoc_target, cfg):
    if assoc_target is None or cfg["lambda_assoc"] == 0:
        return zero_scalar(out.aki_logits.device)
    if cfg["assoc_mode"] == "static":
        return static_association_mse(
            model,
            assoc_target,
            ignore_diag=True,
            directed=cfg["directed_target"],
        )
    return association_mse(
        out.association_hat,
        assoc_target,
        ignore_diag=True,
        directed=cfg["directed_target"],
    )


def lambda_assoc_schedule(epoch: int, total_epochs: int, cfg) -> float:
    base = cfg["lambda_assoc"]
    if not cfg.get("assoc_decay", False):
        return base
    decay_start = int(total_epochs * cfg["assoc_decay_start"])
    if epoch < decay_start:
        return base
    progress = (epoch - decay_start) / max(1, total_epochs - decay_start)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * max(cfg["assoc_decay_floor"], cosine)


def make_masked_recon_batch(batch, cfg, train: bool):
    """
    Create GraphMAE-style pretext mask.

    Only observed, non-AKI feature values are masked. Missingness mask and AKI are
    never perturbed. The model receives a third pretext-mask channel.
    """
    if (
        (not train)
        or (not cfg.get("use_masked_recon", False))
        or cfg.get("mask_recon_rate", 0.0) <= 0
    ):
        return batch, None, None

    observed = batch.x[:, 1] > 0.5
    non_aki = batch.feature_id != cfg["_aki_idx"]
    eligible = observed & non_aki

    rand = torch.rand(batch.x.size(0), device=batch.x.device)
    pretext_mask = eligible & (rand < cfg["mask_recon_rate"])

    if not pretext_mask.any() and eligible.any() and cfg.get("ensure_recon_mask", True):
        eligible_idx = eligible.nonzero(as_tuple=False).view(-1)
        chosen = eligible_idx[
            torch.randint(eligible_idx.numel(), (1,), device=batch.x.device)
        ]
        pretext_mask[chosen] = True

    if not pretext_mask.any():
        return batch, None, None

    recon_target = batch.x[:, 0].detach().clone()
    masked_batch = batch.clone()
    x_masked = batch.x.clone()
    x_masked[pretext_mask, 0] = cfg["mask_recon_value"]
    masked_batch.x = x_masked
    return masked_batch, pretext_mask, recon_target


def masked_feature_recon_loss(out, recon_target, pretext_mask, batch, binary_mask_f):
    if (
        out.recon_pred is None
        or recon_target is None
        or pretext_mask is None
        or not pretext_mask.any()
    ):
        return zero_scalar(out.aki_logits.device)

    pred = out.recon_pred
    target = recon_target[pretext_mask].to(pred.device)
    feature_ids = batch.feature_id[pretext_mask].long()
    is_binary = binary_mask_f.to(pred.device)[feature_ids].bool()

    losses = []
    weights = []

    if is_binary.any():
        # Binary features were not z-scored in preprocess_fold, so targets are 0/1.
        bin_loss = F.binary_cross_entropy_with_logits(
            pred[is_binary],
            target[is_binary].clamp(0.0, 1.0),
            reduction="mean",
        )
        losses.append(bin_loss)
        weights.append(float(is_binary.sum().item()))

    cont_mask = ~is_binary
    if cont_mask.any():
        cont_loss = F.smooth_l1_loss(
            pred[cont_mask], target[cont_mask], reduction="mean"
        )
        losses.append(cont_loss)
        weights.append(float(cont_mask.sum().item()))

    if not losses:
        return zero_scalar(pred.device)

    weight_t = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
    loss_t = torch.stack(losses)
    return (loss_t * weight_t).sum() / weight_t.sum().clamp(min=1.0)


def compute_loss_bundle(
    model,
    batch_for_model,
    clean_batch,
    assoc_target,
    cfg,
    binary_mask_f,
    train: bool,
    epoch: int,
    pretext_mask=None,
    recon_target=None,
):
    out = model(batch_for_model, pretext_mask=pretext_mask)
    assoc_l = _compute_assoc_loss(model, out, assoc_target, cfg)
    aki_l = F.cross_entropy(out.aki_logits, clean_batch.y.view(-1))
    recon_l = masked_feature_recon_loss(
        out, recon_target, pretext_mask, batch_for_model, binary_mask_f
    )
    kl_l = (
        out.kl_loss if out.kl_loss is not None else zero_scalar(out.aki_logits.device)
    )

    la = (
        lambda_assoc_schedule(epoch, cfg["epochs"], cfg)
        if train
        else cfg["lambda_assoc"]
    )
    loss = la * assoc_l + cfg["lambda_aki"] * aki_l
    if cfg["lambda_kl"] > 0:
        loss = loss + cfg["lambda_kl"] * kl_l
    if cfg.get("use_masked_recon", False) and cfg["lambda_recon"] > 0:
        loss = loss + cfg["lambda_recon"] * recon_l

    parts = {
        "loss": loss,
        "assoc_loss": assoc_l,
        "aki_loss": aki_l,
        "kl_loss": kl_l,
        "recon_loss": recon_l,
        "lambda_assoc_eff": la,
    }
    return out, parts


# ══════════════════════════════════════════════════════════════
# FLAG
# ══════════════════════════════════════════════════════════════


def flag_step(
    model, batch, assoc_target, cfg, device, optimizer, binary_mask_f, epoch: int
):
    """FLAG with valid gradient flow. Perturbs value channel only."""
    batch = batch.to(device)
    base_batch, pretext_mask, recon_target = make_masked_recon_batch(
        batch, cfg, train=True
    )

    x_base = base_batch.x.detach()
    observed = x_base[:, 1] > 0.5
    non_aki = base_batch.feature_id != cfg["_aki_idx"]
    value_mask = observed & non_aki
    if pretext_mask is not None:
        value_mask = value_mask & (~pretext_mask)

    delta = torch.zeros_like(x_base, requires_grad=True)
    if cfg.get("flag_random_start", True) and value_mask.any():
        with torch.no_grad():
            delta[:, 0].uniform_(-cfg["flag_eps"], cfg["flag_eps"])
            delta[:, 1].zero_()
            delta[~value_mask] = 0.0

    optimizer.zero_grad(set_to_none=True)
    metric_parts = None
    last_out = None

    for step in range(cfg["flag_steps"]):
        x_adv = x_base + delta
        adv_batch = base_batch.clone()
        adv_batch.x = x_adv

        out, parts = compute_loss_bundle(
            model,
            adv_batch,
            batch,
            assoc_target,
            cfg,
            binary_mask_f,
            train=True,
            epoch=epoch,
            pretext_mask=pretext_mask,
            recon_target=recon_target,
        )
        (parts["loss"] / cfg["flag_steps"]).backward()
        metric_parts = parts
        last_out = out

        if step < cfg["flag_steps"] - 1:
            if delta.grad is None:
                break
            with torch.no_grad():
                delta[:, 0].add_(cfg["flag_eps"] * delta.grad[:, 0].sign())
                delta[:, 0].clamp_(-cfg["flag_eps"], cfg["flag_eps"])
                delta[:, 1].zero_()
                delta[~value_mask] = 0.0
            delta.grad.zero_()

    if cfg["grad_clip"] > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    optimizer.step()
    return last_out, metric_parts


# ══════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════


def run_epoch(
    model,
    loader,
    optimizer,
    assoc_target,
    cfg,
    device,
    train: bool,
    binary_mask_f,
    fgsam=None,
    epoch: int = 0,
):
    model.train(train)
    totals = {
        "loss": 0.0,
        "assoc_loss": 0.0,
        "aki_loss": 0.0,
        "kl_loss": 0.0,
        "recon_loss": 0.0,
    }
    y_true, y_prob = [], []

    for batch in loader:
        batch = batch.to(device)

        if train and cfg.get("use_flag", False):
            out, parts = flag_step(
                model, batch, assoc_target, cfg, device, optimizer, binary_mask_f, epoch
            )
            # For metrics, use the adversarial/pretext forward that produced the step.
        else:
            batch_for_model, pretext_mask, recon_target = make_masked_recon_batch(
                batch, cfg, train=train
            )

            if train:
                optimizer.zero_grad(set_to_none=True)

            out, parts = compute_loss_bundle(
                model,
                batch_for_model,
                batch,
                assoc_target,
                cfg,
                binary_mask_f,
                train=train,
                epoch=epoch,
                pretext_mask=pretext_mask,
                recon_target=recon_target,
            )

            if train and torch.isfinite(parts["loss"]):
                extra_loss = None
                if cfg["lambda_kl"] > 0 or (
                    cfg.get("use_masked_recon", False) and cfg["lambda_recon"] > 0
                ):
                    extra_loss = (
                        cfg["lambda_kl"] * parts["kl_loss"]
                        + cfg["lambda_recon"] * parts["recon_loss"]
                    )

                if cfg.get("use_pcgrad", False) and assoc_target is not None:
                    la = parts["lambda_assoc_eff"]
                    pcgrad_step(
                        model,
                        la * parts["assoc_loss"],
                        cfg["lambda_aki"] * parts["aki_loss"],
                        extra_loss,
                        optimizer,
                        cfg,
                    )
                elif fgsam is not None:
                    parts["loss"].backward()

                    def closure():
                        _, cparts = compute_loss_bundle(
                            model,
                            batch_for_model,
                            batch,
                            assoc_target,
                            cfg,
                            binary_mask_f,
                            train=True,
                            epoch=epoch,
                            pretext_mask=pretext_mask,
                            recon_target=recon_target,
                        )
                        return cparts["loss"]

                    fgsam.step(closure)
                else:
                    parts["loss"].backward()
                    if cfg["grad_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg["grad_clip"]
                        )
                    optimizer.step()

        n = batch.num_graphs
        for k in totals:
            val = parts.get(k)
            if torch.is_tensor(val):
                val = val.detach().item()
            totals[k] += float(val) * n

        probs = out.aki_logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(batch.y.view(-1).detach().cpu().numpy().tolist())

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
        for md in [2, 3, 4, 6]:
            for lr_rate in [0.01, 0.03, 0.05, 0.1]:
                for ss in [0.7, 0.9, 1.0]:
                    xgb = XGBClassifier(
                        n_estimators=1000,
                        max_depth=md,
                        learning_rate=lr_rate,
                        subsample=ss,
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
    config["_aki_idx"] = int(aki_idx)
    num_features = X_nf.shape[1]

    x_scaled, obs_mask = preprocess_fold(X_nf, missing_nf, binary_mask_f, train_idx)
    y = X_nf[:, aki_idx].float().numpy().astype(np.int64)

    edge_index, edge_weight = make_graph_for_fold(config, x_scaled[train_idx], data_dir)
    use_edge_weights = config.get("use_edge_weights", False)
    edge_attr = edge_weight.unsqueeze(-1) if use_edge_weights else None
    edge_dim = 1 if use_edge_weights else None

    assoc_target, is_directed = build_association_target(
        config["assoc_target"],
        x_scaled[train_idx],
        data_dir,
    )
    if assoc_target is not None:
        assoc_target = assoc_target.to(device)
    config["directed_target"] = bool(is_directed)

    # Mask AKI only after building train-fold association targets.
    x_scaled[:, aki_idx] = 0.0
    obs_mask[:, aki_idx] = 0.0

    dataset = FeatureGraphDataset(x_scaled, obs_mask, y, edge_index, edge_attr)
    train_loader = make_loader(dataset, train_idx, config["batch_size"], shuffle=True)
    val_loader = make_loader(dataset, val_idx, config["batch_size"], shuffle=False)
    test_loader = make_loader(dataset, test_idx, config["batch_size"], shuffle=False)

    model = build_model(
        n_features=num_features,
        aki_idx=aki_idx,
        hidden_dim=config["hidden"],
        embedding_dim=config["latent"],
        n_layers=config["layers"],
        encoder_type=config["encoder_type"],
        n_heads=config["heads"],
        dropout=config["dropout"],
        edge_dim=edge_dim,
        decoder_type=config["decoder_type"],
        use_vgae=config["use_vgae"],
        use_laplacian_pe=config["use_laplacian_pe"],
        pe_dim=config.get("pe_dim", 16),
        use_dir_conv=config["use_dir_conv"],
        readout=config["readout"],
        use_masked_recon=config.get("use_masked_recon", False),
    ).to(device)

    if config["use_laplacian_pe"]:
        model.register_laplacian_pe(edge_index)

    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer, T_max=config["epochs"]
    )

    fgsam = None
    if config.get("use_fgsam", False):
        fgsam = FGSAM(
            model.parameters(),
            base_optimizer,
            rho=config["fgsam_rho"],
            grad_clip=config["grad_clip"],
        )

    if config.get("use_pcgrad", False) and config.get("use_fgsam", False):
        print(
            "  note: --use_pcgrad and --use_fgsam are both set; PCGrad takes precedence per batch."
        )

    swa_model, swa_scheduler = None, None
    swa_start = (
        int(config["epochs"] * config["swa_start_frac"])
        if config["use_swa"]
        else config["epochs"] + 1
    )
    if config["use_swa"]:
        from torch.optim.swa_utils import SWALR, AveragedModel

        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(base_optimizer, swa_lr=config["lr"] * 0.5)

    binary_mask_device = binary_mask_f.to(device)
    best_val_auc, best_epoch, best_state = -1.0, -1, None
    patience_left = config["patience"]

    for epoch in range(1, config["epochs"] + 1):
        in_swa = config["use_swa"] and epoch >= swa_start

        train_m = run_epoch(
            model,
            train_loader,
            base_optimizer,
            assoc_target,
            config,
            device,
            train=True,
            binary_mask_f=binary_mask_device,
            fgsam=fgsam,
            epoch=epoch,
        )

        if in_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if epoch == 1 or epoch % config["eval_every"] == 0:
            eval_model = swa_model if (in_swa and swa_model is not None) else model
            val_m = run_epoch(
                eval_model,
                val_loader,
                None,
                assoc_target,
                config,
                device,
                train=False,
                binary_mask_f=binary_mask_device,
                epoch=epoch,
            )

            if config.get("wandb") and HAS_WANDB:
                global_step = (fold - 1) * config["epochs"] + epoch
                wandb.log(
                    {
                        f"fold{fold}/train_loss": train_m["loss"],
                        f"fold{fold}/train_assoc": train_m["assoc_loss"],
                        f"fold{fold}/train_aki": train_m["aki_loss"],
                        f"fold{fold}/train_recon": train_m["recon_loss"],
                        f"fold{fold}/train_auroc": train_m["auroc"],
                        f"fold{fold}/val_loss": val_m["loss"],
                        f"fold{fold}/val_assoc": val_m["assoc_loss"],
                        f"fold{fold}/val_aki": val_m["aki_loss"],
                        f"fold{fold}/val_auroc": val_m["auroc"],
                        f"fold{fold}/val_auprc": val_m["auprc"],
                        f"fold{fold}/lr": base_optimizer.param_groups[0]["lr"],
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            if np.isfinite(val_m["auroc"]) and val_m["auroc"] > best_val_auc:
                best_val_auc = val_m["auroc"]
                best_epoch = epoch
                target_model = (
                    swa_model.module if (in_swa and swa_model is not None) else model
                )
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in target_model.state_dict().items()
                }
                patience_left = config["patience"]
            else:
                patience_left -= 1

            if config.get("verbose"):
                print(
                    f"    ep={epoch:03d} val_auc={val_m['auroc']:.4f} "
                    f"best={best_val_auc:.4f} assoc={val_m['assoc_loss']:.4f} "
                    f"aki={val_m['aki_loss']:.4f} recon={val_m['recon_loss']:.4f} "
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
        assoc_target,
        config,
        device,
        train=False,
        binary_mask_f=binary_mask_device,
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
        "assoc_loss": test_m["assoc_loss"],
        "aki_loss": test_m["aki_loss"],
        "recon_loss": test_m["recon_loss"],
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
    set_seed(config["seed"], deterministic=config.get("deterministic", False))
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
    print(
        f"v7-fixed | readout={config['readout']} assoc_target={config['assoc_target']} "
        f"assoc_mode={config['assoc_mode']} decoder={config['decoder_type']}"
    )
    print(
        f"topology={config['edge_method']} k={config['k']} edge_weights={config['use_edge_weights']} "
        f"dirconv={config['use_dir_conv']} masked_recon={config['use_masked_recon']}"
    )
    print(f"N={X_nf.shape[0]} F={X_nf.shape[1]} AKI={y_all.mean():.3f} device={device}")
    print("=" * 72)

    if config.get("wandb") and HAS_WANDB:
        run_name = (
            config.get("run_name")
            or f"v7fixed_{config['readout']}_{config['assoc_target']}_{config['assoc_mode']}"
        )
        wandb.init(
            project=config.get("wandb_project", "dualr-graph"),
            name=run_name,
            config=config,
            reinit=True,
        )

    if config.get("use_pcgrad", False) and config["assoc_target"] == "none":
        print("note: --use_pcgrad ignored because --assoc_target none.")

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

        print(f"  GNN  AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}")
        print(f"  LR   AUROC={metrics['logreg_auroc']:.4f}")
        if np.isfinite(metrics.get("xgb_auroc", float("nan"))):
            print(f"  XGB  AUROC={metrics['xgb_auroc']:.4f}")

    aurocs = np.array([m["auroc"] for m in fold_metrics])
    lr_aurocs = np.array([m["logreg_auroc"] for m in fold_metrics])
    xgb_aurocs = np.array([m.get("xgb_auroc", float("nan")) for m in fold_metrics])

    summary = {
        "strategy": config.get("run_name") or "v7_fixed",
        "readout": config["readout"],
        "assoc_target": config["assoc_target"],
        "assoc_mode": config["assoc_mode"],
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "auprc_mean": float(np.nanmean([m["auprc"] for m in fold_metrics])),
        "auprc_std": float(np.nanstd([m["auprc"] for m in fold_metrics])),
        "assoc_loss_mean": float(np.nanmean([m["assoc_loss"] for m in fold_metrics])),
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
# CLI
# ══════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="Feature-graph AE v7 fixed")

    # Data
    p.add_argument("--data_dir", default="data")
    p.add_argument("--landmark_days", type=int, default=180)

    # Topology
    p.add_argument("--edge_method", default="llm", choices=["llm", "spearman", "mi"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--use_edge_weights", action="store_true")
    p.add_argument("--cv_cutoff", type=float, default=None)

    # Model
    p.add_argument(
        "--encoder_type", default="gcn", choices=["gcn", "gatv2", "graph_transformer"]
    )
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--decoder_type",
        default="symmetric",
        choices=["symmetric", "directed", "crosscorr"],
    )
    p.add_argument(
        "--readout",
        default="aki_concat",
        choices=["global_pool", "non_aki_pool", "aki_only", "aki_concat"],
    )
    p.add_argument("--use_dir_conv", action="store_true")
    p.add_argument("--use_vgae", action="store_true")
    p.add_argument("--use_laplacian_pe", action="store_true")
    p.add_argument("--pe_dim", type=int, default=16)

    # Masked feature reconstruction
    p.add_argument("--use_masked_recon", action="store_true")
    p.add_argument("--mask_recon_rate", type=float, default=0.20)
    p.add_argument("--mask_recon_value", type=float, default=0.0)
    p.add_argument("--lambda_recon", type=float, default=0.1)
    p.add_argument(
        "--no_ensure_recon_mask", dest="ensure_recon_mask", action="store_false"
    )
    p.set_defaults(ensure_recon_mask=True)

    # Association target, decoupled from topology
    p.add_argument(
        "--assoc_target", default="pearson", choices=["pearson", "llm", "none"]
    )
    p.add_argument("--assoc_mode", default="static", choices=["patient", "static"])

    # Loss
    p.add_argument("--lambda_assoc", type=float, default=1.0)
    p.add_argument("--lambda_aki", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=0.001)
    p.add_argument("--assoc_decay", action="store_true")
    p.add_argument("--assoc_decay_start", type=float, default=0.3)
    p.add_argument("--assoc_decay_floor", type=float, default=0.01)

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=5.0)

    # Enhancements
    p.add_argument("--use_fgsam", action="store_true")
    p.add_argument("--fgsam_rho", type=float, default=0.05)
    p.add_argument("--use_pcgrad", action="store_true")
    p.add_argument("--use_flag", action="store_true")
    p.add_argument("--flag_eps", type=float, default=0.01)
    p.add_argument("--flag_steps", type=int, default=3)
    p.add_argument(
        "--no_flag_random_start", dest="flag_random_start", action="store_false"
    )
    p.set_defaults(flag_random_start=True)
    p.add_argument("--use_swa", action="store_true")
    p.add_argument("--swa_start_frac", type=float, default=0.75)

    # CV / infra
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="dualr-graph")
    p.add_argument("--run_name", default=None)

    return p.parse_args()


def main():
    args = parse_args()
    config = vars(args)
    summary = run_cv(config)
    os.makedirs("results", exist_ok=True)
    name = args.run_name or summary["strategy"]
    with open(os.path.join("results", f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: results/{name}.json")


if __name__ == "__main__":
    main()
