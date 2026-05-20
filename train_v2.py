#!/usr/bin/env python3
"""
train_v2.py — Training with ICML/NeurIPS/ICLR-inspired enhancements.

Upgrades over train.py (v4):
  ① Directed association target from LLM adj (P(i|j) ≠ P(j|i))
  ② FGSAM optimizer (Luo et al., NeurIPS 2024) — flat-minima search
  ③ PCGrad (Yu et al., NeurIPS 2020) — gradient surgery for multi-task
  ④ FLAG (Kong et al., CVPR 2022) — adversarial feature augmentation
  ⑤ SWA (Izmailov et al., UAI 2018) — weight averaging
  ⑥ Directed / cross-correlation decoder (models_v2.py)
  ⑦ Optional VGAE with KL regularisation
  ⑧ Laplacian Positional Encoding

All enhancements are togglable via CLI flags.
Default config reproduces v4.3 baseline for fair comparison.
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
from models_v2 import association_loss, build_model

# ══════════════════════════════════════════════════════════════
#  Training Utilities (FGSAM, PCGrad, FLAG)
# ══════════════════════════════════════════════════════════════


class FGSAM:
    """
    Fast Graph Sharpness-Aware Minimization (Luo et al., NeurIPS 2024).

    Wraps a base optimizer. Each step:
      1. Compute loss and gradient at current weights
      2. Perturb weights: w + ρ * grad / ||grad||
      3. Compute gradient at perturbed weights
      4. Restore weights and apply perturbed gradient
    """

    def __init__(self, params, base_optimizer, rho=0.05):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for p in self.params if p.grad is not None])
        )
        return norm + 1e-12

    @torch.no_grad()
    def _perturb(self):
        norm = self._grad_norm()
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
        """
        closure: callable that returns loss tensor.
        Must be called AFTER the first backward (grads populated).
        """
        # Step 1: perturbation using existing grads
        eps_list = self._perturb()

        # Step 2: recompute loss + grad at perturbed weights
        self.base_optimizer.zero_grad()
        loss = closure()
        loss.backward()

        # Step 3: restore weights, apply perturbed gradient
        self._restore(eps_list)
        self.base_optimizer.step()
        return loss


def pcgrad_project(grad_assoc: list[torch.Tensor], grad_aki: list[torch.Tensor]):
    """
    PCGrad (Yu et al., NeurIPS 2020): project conflicting gradients.

    If cos(g_assoc, g_aki) < 0, remove the conflicting component.
    Modifies gradients in-place.
    """
    # Flatten to vectors
    flat_a = torch.cat([g.flatten() for g in grad_assoc if g is not None])
    flat_b = torch.cat([g.flatten() for g in grad_aki if g is not None])

    dot = torch.dot(flat_a, flat_b)
    if dot >= 0:
        return  # Not conflicting — no surgery needed

    # Project g_a onto normal plane of g_b (and vice versa)
    norm_b_sq = torch.dot(flat_b, flat_b).clamp(min=1e-12)
    norm_a_sq = torch.dot(flat_a, flat_a).clamp(min=1e-12)

    proj_a = dot / norm_b_sq
    proj_b = dot / norm_a_sq

    offset = 0
    for g_a, g_b in zip(grad_assoc, grad_aki):
        if g_a is None or g_b is None:
            continue
        n = g_a.numel()
        g_a.sub_(proj_a * g_b)
        g_b.sub_(proj_b * g_a)
        offset += n


def flag_augment(model, batch, assoc_target, cfg, device, eps=0.01, n_steps=3):
    """
    FLAG (Kong et al., CVPR 2022): adversarial feature augmentation.

    Perturbs node features along the gradient of the loss to generate
    hard examples. Averages loss over clean + perturbed passes.
    """
    batch = batch.to(device)
    x_orig = batch.x.clone()

    total_loss = torch.tensor(0.0, device=device)

    for step in range(n_steps):
        if step == 0:
            # Clean forward pass
            batch.x = x_orig.clone()
            batch.x.requires_grad_(True)
        else:
            # Perturb: x + eps * sign(grad_x) (FGSM-style)
            with torch.no_grad():
                perturbation = eps * batch.x.grad.sign()
                batch.x = (x_orig + perturbation).detach()
                batch.x.requires_grad_(True)

        out = model(batch)
        assoc_l = association_loss(
            out.association_hat,
            assoc_target,
            ignore_diagonal=cfg["ignore_assoc_diagonal"],
            directed=cfg["directed_target"],
        )
        aki_l = F.cross_entropy(out.aki_logits, batch.y.view(-1))
        loss = cfg["lambda_assoc"] * assoc_l + cfg["lambda_aki"] * aki_l
        if out.kl_loss is not None:
            loss = loss + cfg["lambda_kl"] * out.kl_loss

        loss.backward(retain_graph=(step < n_steps - 1))
        total_loss = total_loss + loss.detach()

    batch.x = x_orig  # Restore
    return total_loss / n_steps, out


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════


def set_seed(seed: int):
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


# ══════════════════════════════════════════════════════════════
#  Preprocessing
# ══════════════════════════════════════════════════════════════


def preprocess_fold(X_nf, missing_nf, binary_mask_f, train_idx):
    """Train-only z-score + mean/mode imputation. AKI processed normally."""
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
            fill_z = (fill - mean) / std
            col[missing[:, j]] = fill_z
            x_scaled[:, j] = col.astype(np.float32)

    return x_scaled, obs_mask


def build_association_target(
    edge_method: str,
    x_train_scaled: np.ndarray,
    data_dir: str,
) -> tuple[torch.Tensor, bool]:
    """
    Build the reconstruction target.
    LLM → asymmetric P(i|j) matrix (directed=True)
    Spearman/MI → symmetric |corr| (directed=False)
    """
    if edge_method == "llm":
        adj_path = os.path.join(data_dir, "llm_adj.npy")
        adj = np.load(adj_path).astype(np.float32)
        # Normalise to [0, 1] if not already
        if adj.max() > 1.0:
            adj = adj / adj.max()
        return torch.from_numpy(adj), True
    else:
        corr = np.corrcoef(x_train_scaled, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        assoc = np.abs(corr).astype(np.float32)
        np.fill_diagonal(assoc, 1.0)
        return torch.from_numpy(assoc), False


# ══════════════════════════════════════════════════════════════
#  PyG Dataset
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
#  Graph construction (reuses graph.py)
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
            data_dir=data_dir,
            k=config["k"],
            cv_cutoff=config.get("cv_cutoff"),
        )
    return edge_index.long(), edge_weight.float()


# ══════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════


def run_epoch(
    model,
    loader,
    optimizer,
    assoc_target,
    cfg,
    device,
    train,
    fgsam=None,
    use_pcgrad=False,
    use_flag=False,
):
    model.train(train)
    totals = {"loss": 0.0, "assoc_loss": 0.0, "aki_loss": 0.0, "kl_loss": 0.0}
    y_true, y_prob = [], []

    for batch in loader:
        batch = batch.to(device)

        # ── FLAG augmentation ──
        if train and use_flag:
            avg_loss, out = flag_augment(
                model,
                batch,
                assoc_target,
                cfg,
                device,
                eps=cfg["flag_eps"],
                n_steps=cfg["flag_steps"],
            )
            if optimizer is not None:
                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
            n = batch.num_graphs
            totals["loss"] += avg_loss.item() * n

        else:
            if train:
                optimizer.zero_grad()

            out = model(batch)
            assoc_l = association_loss(
                out.association_hat,
                assoc_target,
                ignore_diagonal=cfg["ignore_assoc_diagonal"],
                directed=cfg["directed_target"],
            )
            aki_l = F.cross_entropy(out.aki_logits, batch.y.view(-1))
            loss = cfg["lambda_assoc"] * assoc_l + cfg["lambda_aki"] * aki_l
            if out.kl_loss is not None:
                loss = loss + cfg["lambda_kl"] * out.kl_loss

            if train and torch.isfinite(loss):
                # ── PCGrad ──
                if use_pcgrad and assoc_l.requires_grad and aki_l.requires_grad:
                    shared = [p for p in model.encoder.parameters() if p.requires_grad]
                    g_assoc = torch.autograd.grad(
                        cfg["lambda_assoc"] * assoc_l,
                        shared,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    g_aki = torch.autograd.grad(
                        cfg["lambda_aki"] * aki_l,
                        shared,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    g_assoc = [
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(g_assoc, shared)
                    ]
                    g_aki = [
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(g_aki, shared)
                    ]
                    pcgrad_project(g_assoc, g_aki)
                    # Apply surgered gradients
                    for p, ga, gb in zip(shared, g_assoc, g_aki):
                        p.grad = ga + gb
                    # Backward for non-shared params (decoder, aki_head)
                    loss.backward()
                    # Overwrite shared grads with surgered ones
                    for p, ga, gb in zip(shared, g_assoc, g_aki):
                        p.grad = ga + gb

                elif fgsam is not None:
                    # FGSAM: first backward already done before calling step
                    loss.backward()

                    def closure():
                        out2 = model(batch)
                        al = association_loss(
                            out2.association_hat,
                            assoc_target,
                            ignore_diagonal=cfg["ignore_assoc_diagonal"],
                            directed=cfg["directed_target"],
                        )
                        ak = F.cross_entropy(out2.aki_logits, batch.y.view(-1))
                        l = cfg["lambda_assoc"] * al + cfg["lambda_aki"] * ak
                        if out2.kl_loss is not None:
                            l = l + cfg["lambda_kl"] * out2.kl_loss
                        return l

                    fgsam.step(closure)

                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()

            n = batch.num_graphs
            totals["loss"] += loss.item() * n
            totals["assoc_loss"] += assoc_l.item() * n
            totals["aki_loss"] += aki_l.item() * n
            if out.kl_loss is not None:
                totals["kl_loss"] += out.kl_loss.item() * n

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
#  Baselines
# ══════════════════════════════════════════════════════════════


def run_baselines(x_scaled, y, train_idx, test_idx, aki_idx, config):
    non_aki = [i for i in range(x_scaled.shape[1]) if i != aki_idx]
    X_bl = x_scaled[:, non_aki]
    lr = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=config.get("logreg_C", 1.0)
    )
    lr.fit(X_bl[train_idx], y[train_idx])
    lr_prob = lr.predict_proba(X_bl[test_idx])[:, 1]
    lr_auc, lr_auprc = safe_auroc(y[test_idx], lr_prob), safe_auprc(
        y[test_idx], lr_prob
    )

    xgb_auc, xgb_auprc = float("nan"), float("nan")
    try:
        from xgboost import XGBClassifier

        n_pos = y[train_idx].sum()
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
        xgb.fit(X_bl[train_idx], y[train_idx])
        xgb_prob = xgb.predict_proba(X_bl[test_idx])[:, 1]
        xgb_auc, xgb_auprc = safe_auroc(y[test_idx], xgb_prob), safe_auprc(
            y[test_idx], xgb_prob
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

    # Preprocess (AKI still present for association target / graph)
    x_scaled, obs_mask = preprocess_fold(X_nf, missing_nf, binary_mask_f, train_idx)
    y = X_nf[:, aki_idx].float().numpy().astype(np.int64)

    # Build graph
    edge_index, edge_weight = make_graph_for_fold(config, x_scaled[train_idx], data_dir)
    use_ew = config.get("use_edge_weights", False)
    edge_attr = edge_weight.unsqueeze(-1) if use_ew else None
    edge_dim = 1 if use_ew else None

    # Build association target (directed for LLM, symmetric for others)
    assoc_target, is_directed = build_association_target(
        config["edge_method"],
        x_scaled[train_idx],
        data_dir,
    )
    assoc_target = assoc_target.to(device)
    config["directed_target"] = is_directed

    # Mask AKI node AFTER building targets
    x_scaled[:, aki_idx] = 0.0
    obs_mask[:, aki_idx] = 0.0

    # Dataset
    dataset = FeatureGraphDataset(x_scaled, obs_mask, y, edge_index, edge_attr)
    train_loader = make_loader(dataset, train_idx, config["batch_size"], shuffle=True)
    val_loader = make_loader(dataset, val_idx, config["batch_size"], shuffle=False)
    test_loader = make_loader(dataset, test_idx, config["batch_size"], shuffle=False)

    # Model
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
    ).to(device)

    # Register LapPE if enabled
    if config["use_laplacian_pe"]:
        model.register_laplacian_pe(edge_index)

    # Optimizer
    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer,
        T_max=config["epochs"],
    )

    fgsam = None
    if config["use_fgsam"]:
        fgsam = FGSAM(model.parameters(), base_optimizer, rho=config["fgsam_rho"])

    # SWA setup
    swa_model = None
    swa_scheduler = None
    swa_start = (
        int(config["epochs"] * config["swa_start_frac"])
        if config["use_swa"]
        else config["epochs"] + 1
    )
    if config["use_swa"]:
        from torch.optim.swa_utils import SWALR, AveragedModel

        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(base_optimizer, swa_lr=config["lr"] * 0.5)

    best_val_auc, best_epoch = -1.0, -1
    best_state = None
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
            fgsam=fgsam,
            use_pcgrad=config["use_pcgrad"],
            use_flag=config["use_flag"],
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
            )

            if config.get("wandb") and HAS_WANDB:
                wandb.log(
                    {
                        f"fold{fold}/train_loss": train_m["loss"],
                        f"fold{fold}/train_assoc": train_m["assoc_loss"],
                        f"fold{fold}/train_aki": train_m["aki_loss"],
                        f"fold{fold}/train_auroc": train_m["auroc"],
                        f"fold{fold}/val_loss": val_m["loss"],
                        f"fold{fold}/val_assoc": val_m["assoc_loss"],
                        f"fold{fold}/val_aki": val_m["aki_loss"],
                        f"fold{fold}/val_auroc": val_m["auroc"],
                        f"fold{fold}/val_auprc": val_m["auprc"],
                        f"fold{fold}/lr": scheduler.get_last_lr()[0],
                        f"fold{fold}/kl_loss": val_m.get("kl_loss", 0),
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            if np.isfinite(val_m["auroc"]) and val_m["auroc"] > best_val_auc:
                best_val_auc = val_m["auroc"]
                best_epoch = epoch
                target_model = swa_model.module if in_swa else model
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in target_model.state_dict().items()
                }
                patience_left = config["patience"]
            else:
                patience_left -= 1

            if config.get("verbose"):
                extras = []
                if config["use_fgsam"]:
                    extras.append("fgsam")
                if config["use_pcgrad"]:
                    extras.append("pcgrad")
                if config["use_flag"]:
                    extras.append("flag")
                if in_swa:
                    extras.append("swa")
                tag = f" [{'+'.join(extras)}]" if extras else ""
                print(
                    f"    ep={epoch:03d} val_auc={val_m['auroc']:.4f} "
                    f"best={best_val_auc:.4f} assoc={val_m['assoc_loss']:.4f} "
                    f"pat={patience_left}{tag}"
                )
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    test_m = run_epoch(
        model, test_loader, None, assoc_target, config, device, train=False
    )
    lr_auc, lr_auprc, xgb_auc, xgb_auprc = run_baselines(
        x_scaled,
        y,
        train_idx,
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
        "best_val_auroc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "logreg_auroc": lr_auc,
        "logreg_auprc": lr_auprc,
        "xgb_auroc": xgb_auc,
        "xgb_auprc": xgb_auprc,
    }


# ══════════════════════════════════════════════════════════════
#  CV loop
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
    eligible_idx = np.where(eligible)[0]
    X_fn = X_fn[:, eligible_idx]
    missing_fn = missing_fn[:, eligible_idx]
    y_all = y_all[eligible_idx]
    X_nf = X_fn.T.contiguous()
    missing_nf = missing_fn.T.contiguous()

    enhancements = []
    if config["decoder_type"] != "symmetric":
        enhancements.append(config["decoder_type"])
    if config["use_fgsam"]:
        enhancements.append("fgsam")
    if config["use_pcgrad"]:
        enhancements.append("pcgrad")
    if config["use_flag"]:
        enhancements.append("flag")
    if config["use_swa"]:
        enhancements.append("swa")
    if config["use_vgae"]:
        enhancements.append("vgae")
    if config["use_laplacian_pe"]:
        enhancements.append("lappe")
    enh_str = "+".join(enhancements) if enhancements else "baseline"

    print("=" * 72)
    print(f"Feature-graph AE v5 | {enh_str}")
    print("=" * 72)
    print(
        f"Device: {device}  N={X_nf.shape[0]}  F={X_nf.shape[1]}  AKI={y_all.mean():.3f}"
    )
    print(
        f"Graph: {config['edge_method']} k={config['k']} conv={config['encoder_type']} L={config['layers']}"
    )
    print(
        f"Decoder: {config['decoder_type']}  λ_assoc={config['lambda_assoc']} λ_aki={config['lambda_aki']}"
    )

    if config.get("wandb") and HAS_WANDB:
        run_name = config.get("run_name") or f"v5_{enh_str}"
        wandb.init(
            project=config.get("wandb_project", "dualr-graph"),
            name=run_name,
            config=config,
            tags=["v5", config["decoder_type"], config["edge_method"]] + enhancements,
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

        print(
            f"  GNN  AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  assoc={metrics['assoc_loss']:.4f}"
        )
        print(f"  LR   AUROC={metrics['logreg_auroc']:.4f}")
        if np.isfinite(metrics.get("xgb_auroc", float("nan"))):
            print(f"  XGB  AUROC={metrics['xgb_auroc']:.4f}")

    aurocs = np.array([m["auroc"] for m in fold_metrics])
    lr_aurocs = np.array([m["logreg_auroc"] for m in fold_metrics])
    xgb_aurocs = np.array([m.get("xgb_auroc", float("nan")) for m in fold_metrics])

    summary = {
        "strategy": f"v5_{enh_str}",
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
        table = wandb.Table(
            columns=[
                "fold",
                "auroc",
                "auprc",
                "assoc_loss",
                "best_epoch",
                "logreg_auroc",
                "logreg_auprc",
                "xgb_auroc",
                "xgb_auprc",
            ]
        )
        for m in fold_metrics:
            table.add_data(
                m["fold"],
                m["auroc"],
                m["auprc"],
                m["assoc_loss"],
                m["best_epoch"],
                m["logreg_auroc"],
                m["logreg_auprc"],
                m.get("xgb_auroc", float("nan")),
                m.get("xgb_auprc", float("nan")),
            )
        wandb.log({"fold_results": table})
        wandb.finish()

    return summary


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="Feature-graph AE v5 (directed decoder + training enhancements)"
    )
    # Data
    p.add_argument("--data_dir", default="data")
    p.add_argument("--landmark_days", type=int, default=180)
    # Graph
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
    # Decoder
    p.add_argument(
        "--decoder_type",
        default="directed",
        choices=["symmetric", "directed", "crosscorr"],
    )
    # Loss
    p.add_argument("--lambda_assoc", type=float, default=1.0)
    p.add_argument("--lambda_aki", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=0.001)
    p.add_argument("--ignore_assoc_diagonal", action="store_true", default=True)
    p.add_argument(
        "--no_ignore_diag", dest="ignore_assoc_diagonal", action="store_false"
    )
    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--logreg_C", type=float, default=1.0)
    # ═══ v5 enhancements (all off by default) ═══
    p.add_argument(
        "--use_fgsam",
        action="store_true",
        help="FGSAM flat-minima optimizer (NeurIPS 2024)",
    )
    p.add_argument("--fgsam_rho", type=float, default=0.05)
    p.add_argument(
        "--use_pcgrad",
        action="store_true",
        help="PCGrad gradient surgery (NeurIPS 2020)",
    )
    p.add_argument(
        "--use_flag",
        action="store_true",
        help="FLAG adversarial augmentation (CVPR 2022)",
    )
    p.add_argument("--flag_eps", type=float, default=0.01)
    p.add_argument("--flag_steps", type=int, default=3)
    p.add_argument(
        "--use_swa", action="store_true", help="SWA weight averaging (UAI 2018)"
    )
    p.add_argument("--swa_start_frac", type=float, default=0.75)
    p.add_argument(
        "--use_vgae", action="store_true", help="Variational GAE with KL regularisation"
    )
    p.add_argument(
        "--use_laplacian_pe", action="store_true", help="Laplacian Positional Encoding"
    )
    p.add_argument("--pe_dim", type=int, default=16)
    # CV
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    # Infra
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
