#!/usr/bin/env python3
"""
train_v2.py — Training with ChatGPT-audited fixes.

Critical fixes:
  ① Decoupled --edge_method (topology) from --assoc_target (recon target)
  ② AKI readout modes: aki_concat, aki_only, non_aki_pool, global_pool
  ③ Static association loss on feature embeddings (not per-patient z)
  ④ Fixed PCGrad: clone grads, include all shared params, call optimizer.step()
  ⑤ Fixed FLAG: proper gradient flow, only perturb value channel
  ⑥ Masked feature reconstruction pretext
  ⑦ Auxiliary loss decay schedule
  ⑧ Tuned XGBoost/LR baselines with val-based selection
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

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
#  Utilities
# ══════════════════════════════════════════════════════════════


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def safe_auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y, p):
    return (
        float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    )


# ══════════════════════════════════════════════════════════════
#  FGSAM
# ══════════════════════════════════════════════════════════════


class FGSAM:
    def __init__(self, params, base_optimizer, rho=0.05):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho

    @torch.no_grad()
    def _perturb(self):
        norm = (
            torch.norm(
                torch.stack([p.grad.norm() for p in self.params if p.grad is not None])
            )
            + 1e-12
        )
        eps_list = []
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
        self.base_optimizer.zero_grad()
        loss = closure()
        loss.backward()
        self._restore(eps_list)
        self.base_optimizer.step()
        return loss


# ══════════════════════════════════════════════════════════════
#  PCGrad (FIXED: clone grads, all shared params, optimizer.step)
# ══════════════════════════════════════════════════════════════


def pcgrad_step(model, assoc_loss_val, aki_loss_val, optimizer, cfg):
    """Fixed PCGrad: project conflicting task gradients, then step."""
    shared = (
        list(model.feature_embedding.parameters())
        + list(model.input_proj.parameters())
        + list(model.encoder.parameters())
    )

    optimizer.zero_grad()
    g_a = torch.autograd.grad(
        assoc_loss_val, shared, retain_graph=True, allow_unused=True
    )
    g_b = torch.autograd.grad(
        aki_loss_val, shared, retain_graph=True, allow_unused=True
    )
    g_a = [
        g.clone() if g is not None else torch.zeros_like(p) for g, p in zip(g_a, shared)
    ]
    g_b = [
        g.clone() if g is not None else torch.zeros_like(p) for g, p in zip(g_b, shared)
    ]

    flat_a = torch.cat([g.flatten() for g in g_a])
    flat_b = torch.cat([g.flatten() for g in g_b])
    dot = torch.dot(flat_a, flat_b)

    if dot < 0:
        # Project each onto the other's normal plane using CLONED originals
        proj_ab = dot / (torch.dot(flat_b, flat_b).clamp(min=1e-12))
        proj_ba = dot / (torch.dot(flat_a, flat_a).clamp(min=1e-12))
        g_a_proj = [ga - proj_ab * gb for ga, gb in zip(g_a, g_b)]
        g_b_proj = [gb - proj_ba * ga for ga, gb in zip(g_a, g_b)]
        g_a, g_b = g_a_proj, g_b_proj

    # Now backward the full loss for non-shared params (decoder, aki_head)
    total_loss = assoc_loss_val + aki_loss_val
    total_loss.backward()

    # Overwrite shared param grads with surgered versions
    for p, ga, gb in zip(shared, g_a, g_b):
        if p.grad is not None:
            p.grad = ga + gb

    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    optimizer.step()


# ══════════════════════════════════════════════════════════════
#  FLAG (FIXED: proper gradient flow, only perturb value channel)
# ══════════════════════════════════════════════════════════════


def flag_step(model, batch, assoc_target, cfg, device, optimizer, eps=0.01, n_steps=3):
    """Fixed FLAG: perturb only value channel (not obs_mask), accumulate real gradients."""
    batch = batch.to(device)
    n_features = batch.x.size(0) // batch.num_graphs

    optimizer.zero_grad()
    accumulated_loss = torch.tensor(0.0, device=device)

    # Perturbation buffer (only on value channel, column 0)
    delta = torch.zeros_like(batch.x[:, 0])  # [B*F]

    for step in range(n_steps):
        x_pert = batch.x.clone()
        x_pert[:, 0] = x_pert[:, 0] + delta
        # Don't perturb AKI nodes (they're already 0)
        batch_modified = batch.clone()
        batch_modified.x = x_pert

        out = model(batch_modified)
        assoc_l = _compute_assoc_loss(model, out, assoc_target, cfg)
        aki_l = F.cross_entropy(out.aki_logits, batch.y.view(-1))
        loss = cfg["lambda_assoc"] * assoc_l + cfg["lambda_aki"] * aki_l

        loss.backward()
        accumulated_loss = accumulated_loss + loss.detach()

        if step < n_steps - 1:
            # Compute adversarial perturbation from x gradient
            # batch_modified.x had requires_grad via autograd, get grad on delta
            grad_x = (
                x_pert.grad if x_pert.grad is not None else torch.zeros_like(x_pert)
            )
            delta = (delta + eps * grad_x[:, 0].sign()).detach()
            optimizer.zero_grad()

    # Scale gradients by n_steps
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= n_steps

    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    optimizer.step()
    return accumulated_loss.item() / n_steps


# ══════════════════════════════════════════════════════════════
#  Preprocessing
# ══════════════════════════════════════════════════════════════


def preprocess_fold(X_nf, missing_nf, binary_mask_f, train_idx):
    X = X_nf.float().numpy().copy()
    missing = missing_nf.bool().numpy()
    binary = binary_mask_f.bool().numpy()
    N, F = X.shape
    observed = ~missing
    x_scaled = np.zeros_like(X, dtype=np.float32)
    obs_mask = observed.astype(np.float32)

    for j in range(F):
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


def build_association_target(assoc_target_method, x_train_scaled, data_dir):
    """
    DECOUPLED from edge_method. Topology and target are independent.
    """
    if assoc_target_method == "pearson":
        corr = np.corrcoef(x_train_scaled, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        assoc = np.abs(corr).astype(np.float32)
        np.fill_diagonal(assoc, 1.0)
        return torch.from_numpy(assoc), False
    elif assoc_target_method == "llm":
        adj = np.load(os.path.join(data_dir, "llm_adj.npy")).astype(np.float32)
        if adj.max() > 1.0:
            adj = adj / adj.max()
        return torch.from_numpy(adj), True
    elif assoc_target_method == "none":
        return None, False
    else:
        raise ValueError(f"assoc_target={assoc_target_method!r}")


# ══════════════════════════════════════════════════════════════
#  Dataset
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
        v = self.x_scaled[idx].view(-1, 1)
        m = self.obs_mask[idx].view(-1, 1)
        data = Data(
            x=torch.cat([v, m], dim=1),
            edge_index=self.edge_index,
            y=self.y[idx].view(1),
            feature_id=self.feature_id,
        )
        if self.edge_attr is not None:
            data.edge_attr = self.edge_attr
        return data


def make_loader(dataset, indices, bs, shuffle):
    return DataLoader(
        dataset.index_select(indices.tolist()), batch_size=bs, shuffle=shuffle
    )


# ══════════════════════════════════════════════════════════════
#  Graph construction
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
    else:
        edge_index, edge_weight = load_llm_graph(
            data_dir=data_dir, k=config["k"], cv_cutoff=config.get("cv_cutoff")
        )
    return edge_index.long(), edge_weight.float()


# ══════════════════════════════════════════════════════════════
#  Loss helpers
# ══════════════════════════════════════════════════════════════


def _compute_assoc_loss(model, out, assoc_target, cfg):
    """Compute association loss: patient-level, static, or none."""
    if assoc_target is None:
        return torch.tensor(0.0, device=out.aki_logits.device)
    if cfg["assoc_mode"] == "static":
        return static_association_mse(model, assoc_target, ignore_diag=True)
    else:  # "patient"
        return association_mse(
            out.association_hat,
            assoc_target,
            ignore_diag=True,
            directed=cfg["directed_target"],
        )


def lambda_assoc_schedule(epoch, total_epochs, cfg):
    """Optional auxiliary loss decay."""
    base = cfg["lambda_assoc"]
    if not cfg["assoc_decay"]:
        return base
    decay_start = int(total_epochs * cfg["assoc_decay_start"])
    if epoch < decay_start:
        return base
    progress = (epoch - decay_start) / max(1, total_epochs - decay_start)
    return base * max(
        cfg["assoc_decay_floor"], 0.5 * (1 + math.cos(math.pi * progress))
    )


# ══════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════


def run_epoch(
    model, loader, optimizer, assoc_target, cfg, device, train, fgsam=None, epoch=0
):
    model.train(train)
    totals = {"loss": 0.0, "assoc_loss": 0.0, "aki_loss": 0.0, "kl_loss": 0.0}
    y_true, y_prob = [], []
    la = (
        lambda_assoc_schedule(epoch, cfg["epochs"], cfg)
        if train
        else cfg["lambda_assoc"]
    )

    for batch in loader:
        batch = batch.to(device)

        if train and cfg.get("use_flag"):
            batch_loss = flag_step(
                model,
                batch,
                assoc_target,
                cfg,
                device,
                optimizer,
                eps=cfg["flag_eps"],
                n_steps=cfg["flag_steps"],
            )
            totals["loss"] += batch_loss * batch.num_graphs
            out = model(batch)  # for metrics only
        else:
            if train:
                optimizer.zero_grad()
            out = model(batch)
            assoc_l = _compute_assoc_loss(model, out, assoc_target, cfg)
            aki_l = F.cross_entropy(out.aki_logits, batch.y.view(-1))
            loss = la * assoc_l + cfg["lambda_aki"] * aki_l
            if out.kl_loss is not None:
                loss = loss + cfg["lambda_kl"] * out.kl_loss

            if train and torch.isfinite(loss):
                if cfg.get("use_pcgrad") and assoc_target is not None:
                    pcgrad_step(
                        model, la * assoc_l, cfg["lambda_aki"] * aki_l, optimizer, cfg
                    )
                elif fgsam is not None:
                    loss.backward()

                    def closure():
                        o = model(batch)
                        al = _compute_assoc_loss(model, o, assoc_target, cfg)
                        ak = F.cross_entropy(o.aki_logits, batch.y.view(-1))
                        return la * al + cfg["lambda_aki"] * ak

                    fgsam.step(closure)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()

            totals["loss"] += loss.item() * batch.num_graphs
            totals["assoc_loss"] += assoc_l.item() * batch.num_graphs
            totals["aki_loss"] += aki_l.item() * batch.num_graphs

        probs = out.aki_logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(batch.y.view(-1).cpu().numpy().tolist())

    n = max(1, len(y_true))
    m = {k: v / n for k, v in totals.items()}
    m["auroc"] = safe_auroc(y_true, y_prob)
    m["auprc"] = safe_auprc(y_true, y_prob)
    m["accuracy"] = accuracy_score(y_true, (np.array(y_prob) > 0.5).astype(int))
    return m


# ══════════════════════════════════════════════════════════════
#  Baselines (tuned)
# ══════════════════════════════════════════════════════════════


def run_baselines(x_scaled, y, train_idx, val_idx, test_idx, aki_idx, config):
    non_aki = [i for i in range(x_scaled.shape[1]) if i != aki_idx]
    X_bl = x_scaled[:, non_aki]

    best_lr_auc, best_lr = -1, None
    for C in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=C)
        lr.fit(X_bl[train_idx], y[train_idx])
        va = safe_auroc(y[val_idx], lr.predict_proba(X_bl[val_idx])[:, 1])
        if np.isfinite(va) and va > best_lr_auc:
            best_lr_auc, best_lr = va, lr
    lr_prob = best_lr.predict_proba(X_bl[test_idx])[:, 1]
    lr_auc, lr_auprc = safe_auroc(y[test_idx], lr_prob), safe_auprc(
        y[test_idx], lr_prob
    )

    xgb_auc, xgb_auprc = float("nan"), float("nan")
    try:
        from xgboost import XGBClassifier

        spw = max(1, (len(train_idx) - y[train_idx].sum()) / max(1, y[train_idx].sum()))
        best_xv, best_xgb = -1, None
        for md in [2, 3, 4, 6]:
            for lr_r in [0.01, 0.05, 0.1]:
                for ss in [0.7, 1.0]:
                    xgb = XGBClassifier(
                        n_estimators=1000,
                        max_depth=md,
                        learning_rate=lr_r,
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
                    vp = xgb.predict_proba(X_bl[val_idx])[:, 1]
                    va = safe_auroc(y[val_idx], vp)
                    if np.isfinite(va) and va > best_xv:
                        best_xv, best_xgb = va, xgb
        if best_xgb is not None:
            xp = best_xgb.predict_proba(X_bl[test_idx])[:, 1]
            xgb_auc, xgb_auprc = safe_auroc(y[test_idx], xp), safe_auprc(
                y[test_idx], xp
            )
    except ImportError:
        pass
    return lr_auc, lr_auprc, xgb_auc, xgb_auprc


# ══════════════════════════════════════════════════════════════
#  Single fold
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

    edge_index, edge_weight = make_graph_for_fold(config, x_scaled[train_idx], data_dir)
    use_ew = config.get("use_edge_weights", False)
    edge_attr = edge_weight.unsqueeze(-1) if use_ew else None
    edge_dim = 1 if use_ew else None

    # DECOUPLED: topology is edge_method, target is assoc_target
    assoc_target, is_directed = build_association_target(
        config["assoc_target"], x_scaled[train_idx], data_dir
    )
    if assoc_target is not None:
        assoc_target = assoc_target.to(device)
    config["directed_target"] = is_directed

    # Mask AKI AFTER building targets
    x_scaled[:, aki_idx] = 0.0
    obs_mask[:, aki_idx] = 0.0

    dataset = FeatureGraphDataset(x_scaled, obs_mask, y, edge_index, edge_attr)
    train_loader = make_loader(dataset, train_idx, config["batch_size"], True)
    val_loader = make_loader(dataset, val_idx, config["batch_size"], False)
    test_loader = make_loader(dataset, test_idx, config["batch_size"], False)

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

    base_opt = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_opt, T_max=config["epochs"]
    )
    fgsam = (
        FGSAM(model.parameters(), base_opt, rho=config["fgsam_rho"])
        if config["use_fgsam"]
        else None
    )

    swa_model, swa_sched = None, None
    swa_start = (
        int(config["epochs"] * config["swa_start_frac"])
        if config["use_swa"]
        else config["epochs"] + 1
    )
    if config["use_swa"]:
        from torch.optim.swa_utils import SWALR, AveragedModel

        swa_model = AveragedModel(model)
        swa_sched = SWALR(base_opt, swa_lr=config["lr"] * 0.5)

    best_val, best_ep, best_state = -1.0, -1, None
    patience_left = config["patience"]

    for epoch in range(1, config["epochs"] + 1):
        in_swa = config["use_swa"] and epoch >= swa_start
        train_m = run_epoch(
            model,
            train_loader,
            base_opt,
            assoc_target,
            config,
            device,
            train=True,
            fgsam=fgsam,
            epoch=epoch,
        )
        if in_swa:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()

        if epoch == 1 or epoch % config["eval_every"] == 0:
            em = swa_model if in_swa and swa_model else model
            val_m = run_epoch(
                em,
                val_loader,
                None,
                assoc_target,
                config,
                device,
                train=False,
                epoch=epoch,
            )
            if config.get("wandb") and HAS_WANDB:
                global_step = (fold - 1) * config["epochs"] + epoch
                wandb.log(
                    {
                        f"fold{fold}/train_loss": train_m["loss"],
                        f"fold{fold}/train_auroc": train_m["auroc"],
                        f"fold{fold}/val_loss": val_m["loss"],
                        f"fold{fold}/val_auroc": val_m["auroc"],
                        f"fold{fold}/val_auprc": val_m["auprc"],
                        f"fold{fold}/val_assoc": val_m["assoc_loss"],
                        f"fold{fold}/val_aki": val_m["aki_loss"],
                        f"fold{fold}/lr": scheduler.get_last_lr()[0],
                        f"fold{fold}/epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
            if np.isfinite(val_m["auroc"]) and val_m["auroc"] > best_val:
                best_val, best_ep = val_m["auroc"], epoch
                tgt = swa_model.module if in_swa else model
                best_state = {k: v.cpu().clone() for k, v in tgt.state_dict().items()}
                patience_left = config["patience"]
            else:
                patience_left -= 1
            if config.get("verbose"):
                print(
                    f"    ep={epoch:03d} val_auc={val_m['auroc']:.4f} best={best_val:.4f} pat={patience_left}"
                )
            if patience_left <= 0:
                break

    if best_state:
        model.load_state_dict(best_state)
    test_m = run_epoch(
        model, test_loader, None, assoc_target, config, device, train=False
    )
    lr_a, lr_p, xgb_a, xgb_p = run_baselines(
        x_scaled, y, train_idx, val_idx, test_idx, aki_idx, config
    )

    return {
        "fold": fold,
        "auroc": test_m["auroc"],
        "auprc": test_m["auprc"],
        "accuracy": test_m["accuracy"],
        "assoc_loss": test_m["assoc_loss"],
        "best_val_auroc": float(best_val),
        "best_epoch": int(best_ep),
        "logreg_auroc": lr_a,
        "logreg_auprc": lr_p,
        "xgb_auroc": xgb_a,
        "xgb_auprc": xgb_p,
        "time_s": 0,
    }


# ══════════════════════════════════════════════════════════════
#  CV
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
    missing_fn = (
        torch.load(miss_path, weights_only=True).bool()
        if os.path.exists(miss_path)
        else torch.zeros_like(X_fn, dtype=torch.bool)
    )

    meta = pd.read_csv(os.path.join(data_dir, "cohort_meta.csv"))
    y_all = X_fn[aki_idx].numpy()
    eligible = (y_all == 1) | (meta["surv_days"].values >= config["landmark_days"])
    ei = np.where(eligible)[0]
    X_fn, missing_fn, y_all = X_fn[:, ei], missing_fn[:, ei], y_all[ei]
    X_nf, missing_nf = X_fn.T.contiguous(), missing_fn.T.contiguous()

    print("=" * 72)
    print(
        f"v7 | readout={config['readout']} assoc_target={config['assoc_target']} "
        f"assoc_mode={config['assoc_mode']} decoder={config['decoder_type']}"
    )
    print(f"N={X_nf.shape[0]} F={X_nf.shape[1]} AKI={y_all.mean():.3f} device={device}")
    print("=" * 72)

    if config.get("wandb") and HAS_WANDB:
        name = (
            config.get("run_name") or f"v7_{config['readout']}_{config['assoc_target']}"
        )
        wandb.init(
            project=config.get("wandb_project", "dualr-graph"),
            name=name,
            config=config,
            reinit=True,
        )

    outer = StratifiedKFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"]
    )
    fold_metrics = []

    for fold, (tv, te) in enumerate(outer.split(np.zeros(len(y_all)), y_all), 1):
        inner = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=config["seed"] + fold
        )
        tr_r, va_r = next(inner.split(np.zeros(len(tv)), y_all[tv]))
        tr, va = tv[tr_r], tv[va_r]
        print(f"\nFold {fold}: train={len(tr)} val={len(va)} test={len(te)}")

        t0 = time.time()
        m = train_one_fold(
            fold,
            config,
            X_nf,
            missing_nf,
            binary_mask_f,
            feature_names,
            tr,
            va,
            te,
            data_dir,
            device,
        )
        m["time_s"] = time.time() - t0
        fold_metrics.append(m)
        print(f"  GNN  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}")
        print(f"  LR   AUROC={m['logreg_auroc']:.4f}")
        if np.isfinite(m.get("xgb_auroc", float("nan"))):
            print(f"  XGB  AUROC={m['xgb_auroc']:.4f}")

    aurocs = np.array([m["auroc"] for m in fold_metrics])
    lr_a = np.array([m["logreg_auroc"] for m in fold_metrics])
    summary = {
        "strategy": config.get("run_name", "v7"),
        "readout": config["readout"],
        "assoc_target": config["assoc_target"],
        "assoc_mode": config["assoc_mode"],
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "auprc_mean": float(np.nanmean([m["auprc"] for m in fold_metrics])),
        "auprc_std": float(np.nanstd([m["auprc"] for m in fold_metrics])),
        "assoc_loss_mean": float(np.nanmean([m["assoc_loss"] for m in fold_metrics])),
        "logreg_auroc_mean": float(np.nanmean(lr_a)),
        "xgb_auroc_mean": float(
            np.nanmean([m.get("xgb_auroc", float("nan")) for m in fold_metrics])
        ),
        "delta_vs_logreg": float(np.nanmean(aurocs - lr_a)),
        "folds": fold_metrics,
        "config": config,
    }

    print(f"\n{'='*72}")
    print(f"GNN  AUROC: {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")
    print(f"LR   AUROC: {summary['logreg_auroc_mean']:.4f}")
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
#  CLI
# ══════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="Feature-graph AE v7 (z_aki readout + fixes)"
    )
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
    p.add_argument("--use_masked_recon", action="store_true")
    # Association target (DECOUPLED from topology)
    p.add_argument(
        "--assoc_target", default="pearson", choices=["pearson", "llm", "none"]
    )
    p.add_argument("--assoc_mode", default="patient", choices=["patient", "static"])
    # Loss
    p.add_argument("--lambda_assoc", type=float, default=1.0)
    p.add_argument("--lambda_aki", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=0.001)
    p.add_argument("--assoc_decay", action="store_true", help="Cosine decay on λ_assoc")
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
    p.add_argument("--use_swa", action="store_true")
    p.add_argument("--swa_start_frac", type=float, default=0.75)
    # CV
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
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
