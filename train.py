#!/usr/bin/env python3
"""
train.py — Training with association reconstruction objective.

Key changes from train.py (v3.1):
  - Reconstruction target: feature-feature association matrix (|corr|)
    instead of per-patient feature values
  - Inner-product decoder (parameter-free) replaces GNN decoder
  - No label-as-input — AKI node gets same treatment as other features
  - PyG DataLoader for clean batching
  - Simplified loss: λ_assoc * MSE(A_hat, A_target) + CE(logits, y)

Keeps: graph.py edge methods, CV infrastructure, baselines, wandb.
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
from models import association_mse_loss, build_model

# ──────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
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


# ──────────────────────────────────────────────────────────────
#  Preprocessing (simplified — no label-as-input)
# ──────────────────────────────────────────────────────────────


def preprocess_fold(
    X_nf: torch.Tensor,
    missing_nf: torch.Tensor,
    binary_mask_f: torch.Tensor,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train-only z-score + mean/mode imputation.

    Returns:
        x_scaled   [N, F]  float32 — standardised, imputed values
        obs_mask   [N, F]  float32 — 1 if observed, 0 if imputed
        means      [F]
        stds       [F]
    """
    X = X_nf.float().numpy().copy()
    missing = missing_nf.bool().numpy()
    binary = binary_mask_f.bool().numpy()
    N, F = X.shape
    observed = ~missing

    # Train-only statistics
    x_scaled = np.zeros_like(X, dtype=np.float32)
    obs_mask = observed.astype(np.float32)

    for j in range(F):
        train_obs = train_idx[observed[train_idx, j]]
        vals = X[train_obs, j].astype(np.float64)

        if binary[j]:
            # Mode imputation, no scaling
            fill = float(np.round(vals.mean())) if len(vals) > 0 else 0.0
            col = X[:, j].copy()
            col[missing[:, j]] = fill
            x_scaled[:, j] = col.astype(np.float32)
        else:
            # z-score with train stats
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


def build_association_target(x_train_scaled: np.ndarray) -> torch.Tensor:
    """Absolute Pearson correlation from train set — the reconstruction target."""
    corr = np.corrcoef(x_train_scaled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    assoc = np.abs(corr).astype(np.float32)
    np.fill_diagonal(assoc, 1.0)
    return torch.from_numpy(assoc)


# ──────────────────────────────────────────────────────────────
#  PyG Dataset
# ──────────────────────────────────────────────────────────────


class FeatureGraphDataset(Dataset):
    """One graph per patient. Nodes = features, node input = [value, obs_mask]."""

    def __init__(
        self,
        x_scaled: np.ndarray,
        obs_mask: np.ndarray,
        y: np.ndarray,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ):
        super().__init__()
        self.x_scaled = torch.from_numpy(x_scaled).float()
        self.obs_mask = torch.from_numpy(obs_mask).float()
        self.y = torch.from_numpy(y).long()
        self.edge_index = edge_index.long()
        self.edge_attr = edge_attr
        self.n_features = x_scaled.shape[1]
        self.feature_id = torch.arange(self.n_features, dtype=torch.long)

    def len(self) -> int:
        return self.x_scaled.shape[0]

    def get(self, idx: int) -> Data:
        values = self.x_scaled[idx].view(-1, 1)
        mask = self.obs_mask[idx].view(-1, 1)
        node_x = torch.cat([values, mask], dim=1)  # [F, 2]
        data = Data(
            x=node_x,
            edge_index=self.edge_index,
            y=self.y[idx].view(1),
            feature_id=self.feature_id,
        )
        if self.edge_attr is not None:
            data.edge_attr = self.edge_attr
        return data


# ──────────────────────────────────────────────────────────────
#  Graph construction (reuses graph.py)
# ──────────────────────────────────────────────────────────────


def make_graph_for_fold(config, x_train_scaled, data_dir):
    """Build graph from train set. Returns edge_index, edge_weight."""
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


# ──────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────


def make_loader(dataset, indices, batch_size: int, shuffle: bool) -> DataLoader:
    subset = dataset.index_select(indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, assoc_target, cfg, device, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "assoc_loss": 0.0, "aki_loss": 0.0}
    y_true, y_prob = [], []

    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()

        out = model(batch)

        assoc_loss = association_mse_loss(
            out.association_hat,
            assoc_target,
            ignore_diagonal=cfg["ignore_assoc_diagonal"],
        )
        aki_loss = F.cross_entropy(out.aki_logits, batch.y.view(-1))
        loss = cfg["lambda_assoc"] * assoc_loss + cfg["lambda_aki"] * aki_loss

        if train and torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

        n = batch.num_graphs
        totals["loss"] += loss.item() * n
        totals["assoc_loss"] += assoc_loss.item() * n
        totals["aki_loss"] += aki_loss.item() * n

        probs = out.aki_logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(batch.y.view(-1).detach().cpu().numpy().tolist())

    n_total = max(1, len(y_true))
    metrics = {k: v / n_total for k, v in totals.items()}
    metrics["accuracy"] = accuracy_score(y_true, (np.array(y_prob) > 0.5).astype(int))
    metrics["auroc"] = safe_auroc(y_true, y_prob)
    metrics["auprc"] = safe_auprc(y_true, y_prob)
    return metrics


# ──────────────────────────────────────────────────────────────
#  Baselines
# ──────────────────────────────────────────────────────────────


def run_baselines(x_scaled, y, train_idx, test_idx, aki_idx, config):
    non_aki = [i for i in range(x_scaled.shape[1]) if i != aki_idx]
    X_bl = x_scaled[:, non_aki]
    lr = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=config.get("logreg_C", 1.0)
    )
    lr.fit(X_bl[train_idx], y[train_idx])
    lr_prob = lr.predict_proba(X_bl[test_idx])[:, 1]
    lr_auc = safe_auroc(y[test_idx], lr_prob)
    lr_auprc = safe_auprc(y[test_idx], lr_prob)

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
        xgb_auc = safe_auroc(y[test_idx], xgb_prob)
        xgb_auprc = safe_auprc(y[test_idx], xgb_prob)
    except ImportError:
        pass
    return lr_auc, lr_auprc, xgb_auc, xgb_auprc


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

    # Preprocess
    x_scaled, obs_mask = preprocess_fold(X_nf, missing_nf, binary_mask_f, train_idx)
    y = X_nf[:, aki_idx].float().numpy().astype(np.int64)

    # Build graph from train set
    edge_index, edge_weight = make_graph_for_fold(config, x_scaled[train_idx], data_dir)
    use_ew = config.get("use_edge_weights", False)
    edge_attr = edge_weight.unsqueeze(-1) if use_ew else None
    edge_dim = 1 if use_ew else None

    # Build association target from train set
    assoc_target = build_association_target(x_scaled[train_idx]).to(device)

    # Create PyG dataset
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
        aki_mode=config["aki_mode"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
    )

    best_val_auc, best_epoch = -1.0, -1
    best_state = None
    patience_left = config["patience"]

    for epoch in range(1, config["epochs"] + 1):
        train_m = run_epoch(
            model, train_loader, optimizer, assoc_target, config, device, train=True
        )
        scheduler.step()

        if epoch == 1 or epoch % config["eval_every"] == 0:
            val_m = run_epoch(
                model, val_loader, None, assoc_target, config, device, train=False
            )

            if config.get("wandb") and HAS_WANDB:
                wandb.log(
                    {
                        f"fold{fold}/train_loss": train_m["loss"],
                        f"fold{fold}/train_assoc": train_m["assoc_loss"],
                        f"fold{fold}/train_aki": train_m["aki_loss"],
                        f"fold{fold}/val_auroc": val_m["auroc"],
                        f"fold{fold}/val_assoc": val_m["assoc_loss"],
                        "epoch": epoch,
                    },
                    step=epoch,
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
                    f"best={best_val_auc:.4f} assoc={val_m['assoc_loss']:.4f} pat={patience_left}"
                )
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    test_m = run_epoch(
        model, test_loader, None, assoc_target, config, device, train=False
    )

    # Baselines
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


# ──────────────────────────────────────────────────────────────
#  CV loop
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
    print(f"Feature-graph AE v4 (association decoder) | aki_mode={config['aki_mode']}")
    print("=" * 72)
    print(
        f"Device: {device}  N={X_nf.shape[0]}  F={X_nf.shape[1]}  AKI={y_all.mean():.3f}"
    )
    print(f"Features: {n_bin} binary + {n_cont} continuous")
    print(
        f"Graph: {config['edge_method']} k={config['k']} conv={config['encoder_type']} L={config['layers']}"
    )
    print(f"Loss: λ_assoc={config['lambda_assoc']} λ_aki={config['lambda_aki']}")

    if config.get("wandb") and HAS_WANDB:
        run_name = config.get("run_name") or (
            f"v4_{config['aki_mode']}_{config['encoder_type']}_"
            f"{config['edge_method']}_l{config['layers']}_la{config['lambda_assoc']:g}"
        )
        wandb.init(
            project=config.get("wandb_project", "dualr-graph"),
            name=run_name,
            config=config,
            tags=[
                "v4",
                config["aki_mode"],
                config["edge_method"],
                config["encoder_type"],
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
        print(
            f"  GNN  AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  assoc={metrics['assoc_loss']:.4f}"
        )
        print(f"  LR   AUROC={metrics['logreg_auroc']:.4f}")
        if has_xgb:
            print(f"  XGB  AUROC={metrics['xgb_auroc']:.4f}")

    aurocs = np.array([m["auroc"] for m in fold_metrics])
    lr_aurocs = np.array([m["logreg_auroc"] for m in fold_metrics])
    xgb_aurocs = np.array([m.get("xgb_auroc", float("nan")) for m in fold_metrics])

    summary = {
        "strategy": "v4_association",
        "aki_mode": config["aki_mode"],
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "auprc_mean": float(np.nanmean([m["auprc"] for m in fold_metrics])),
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
    if "delta_vs_xgb" in summary:
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


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Feature-graph AE v4 (association decoder)")
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
        "--encoder_type",
        default="gatv2",
        choices=["gcn", "gat", "gatv2", "graph_transformer"],
    )
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--aki_mode", default="pooled", choices=["pooled", "dual"])
    # Loss
    p.add_argument("--lambda_assoc", type=float, default=1.0)
    p.add_argument("--lambda_aki", type=float, default=1.0)
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
    name = args.run_name or (
        f"v4_{args.aki_mode}_{args.encoder_type}_{args.edge_method}_"
        f"l{args.layers}_la{args.lambda_assoc:g}"
    )
    with open(os.path.join("results", f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: results/{name}.json")


if __name__ == "__main__":
    main()
