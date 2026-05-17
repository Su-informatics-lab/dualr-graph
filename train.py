#!/usr/bin/env python3
"""
train.py — Training loop for Graph Transformer AE.

Handles:
  - Semi-supervised loss (reconstruction + λ·CE on AKI node)
  - Stratified k-fold CV
  - Early stopping on validation AUROC
  - W&B logging
  - Graph rebuilding per fold (from training patients only)

FIX (2026-05-16): The model sees ALL patients at once (transductive).
Train/test split is over patient *indices* (columns), not separate tensors.
in_channels = N (full cohort), constant across folds.

Usage:
    python train.py --arch transformer --edge_method spearman --k 8 \
        --hidden 64 --latent 16 --heads 4 --lr 1e-3 --lambda_ce 1.0 \
        --epochs 300 --cv_folds 5

    # With W&B
    python train.py --arch transformer --edge_method spearman --k 8 \
        --hidden 64 --latent 16 --wandb
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from models import build_model

# ── Loss function ─────────────────────────────────────────────────


def compute_loss(x, x_recon, aki_idx, y_aki, train_idx, lam=1.0, binary_mask=None):
    """
    Semi-supervised loss: reconstruction (all patients) + λ·CE on AKI node
    (training patients only).

    Args:
        x:           [F, N] original feature matrix (ALL patients)
        x_recon:     [F, N] reconstructed (ALL patients)
        aki_idx:     int, index of AKI node
        y_aki:       [N] ground truth (0/1)
        train_idx:   array of int, training patient indices for CE
        lam:         CE weight
        binary_mask: [F] bool, True for binary nodes (BCE), False (MSE)

    Returns:
        total_loss, recon_loss, ce_loss
    """
    # Reconstruction loss — ALL features EXCEPT AKI row
    # (AKI row is masked for test patients, so reconstruction would
    #  train toward 0. AKI prediction comes from CE loss only.)
    recon_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    recon_mask[aki_idx] = False

    if binary_mask is not None:
        cont = (~binary_mask) & recon_mask
        bbin = binary_mask & recon_mask
        n_cont = cont.sum().item()
        n_bin = bbin.sum().item()

        loss_parts = []
        if n_cont > 0:
            loss_parts.append(F.mse_loss(x_recon[cont], x[cont]))
        if n_bin > 0:
            loss_parts.append(
                F.binary_cross_entropy_with_logits(x_recon[bbin], x[bbin])
            )
        recon_loss = sum(loss_parts) / max(len(loss_parts), 1)
    else:
        recon_loss = F.mse_loss(x_recon[recon_mask], x[recon_mask])

    # CE loss on AKI node — TRAINING patients only
    aki_logits = x_recon[aki_idx]  # [N]
    ce_loss = F.binary_cross_entropy_with_logits(
        aki_logits[train_idx], y_aki[train_idx]
    )

    total = recon_loss + lam * ce_loss
    return total, recon_loss, ce_loss


# ── Evaluation ────────────────────────────────────────────────────


def evaluate(model, x, edge_index, aki_idx, y_all, test_idx):
    """Compute AUROC, AUPRC, Brier on held-out patient indices."""
    model.eval()
    with torch.no_grad():
        x_recon, _ = model(x, edge_index)
        probs = torch.sigmoid(x_recon[aki_idx]).cpu().numpy()

    y = y_all.cpu().numpy()
    y_v = y[test_idx]
    p_v = probs[test_idx]

    if len(np.unique(y_v)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}

    return {
        "auroc": roc_auc_score(y_v, p_v),
        "auprc": average_precision_score(y_v, p_v),
        "brier": brier_score_loss(y_v, p_v),
    }


# ── Single fold training ─────────────────────────────────────────


def train_one_fold(
    model,
    x_full,
    edge_index,
    aki_idx,
    y_all,
    train_idx,
    test_idx,
    binary_mask,
    config,
    fold_idx=0,
    wandb_run=None,
):
    """Train one fold, return best metrics dict."""
    device = x_full.device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    # ── Mask test AKI labels to prevent leakage ──
    # The model must INFER test AKI values from other features via the graph.
    x_masked = x_full.clone()
    x_masked[aki_idx, test_idx] = 0.0  # hide test AKI from input

    best_auroc = 0.0
    best_metrics = {}
    patience_counter = 0
    best_state = None

    for epoch in range(config["epochs"]):
        model.train()
        x_recon, z = model(x_masked, edge_index)

        loss, recon_l, ce_l = compute_loss(
            x_masked,
            x_recon,
            aki_idx,
            y_all,
            train_idx,
            lam=config["lambda_ce"],
            binary_mask=binary_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.get("grad_clip", 5.0)
        )
        optimizer.step()
        scheduler.step()

        # Evaluate periodically
        if epoch % config.get("eval_every", 10) == 0:
            metrics = evaluate(model, x_masked, edge_index, aki_idx, y_all, test_idx)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"fold{fold_idx}/train_loss": loss.item(),
                        f"fold{fold_idx}/recon_loss": recon_l.item(),
                        f"fold{fold_idx}/ce_loss": ce_l.item(),
                        f"fold{fold_idx}/val_auroc": metrics["auroc"],
                        f"fold{fold_idx}/val_auprc": metrics["auprc"],
                        f"fold{fold_idx}/epoch": epoch,
                        f"fold{fold_idx}/lr": scheduler.get_last_lr()[0],
                    }
                )

            if not np.isnan(metrics["auroc"]) and metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                best_metrics = metrics.copy()
                best_metrics["best_epoch"] = epoch
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= config.get("patience", 30):
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_metrics


# ── Cross-validation driver ───────────────────────────────────────


def run_cv(config, data_dir="data", wandb_run=None):
    """Run stratified k-fold CV and return fold-level metrics."""
    from graph import build_mi, build_spearman, knn_sparsify

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X = torch.load(
        os.path.join(data_dir, "feature_matrix.pt"), weights_only=True
    )  # [F, N]
    binary_mask = torch.load(
        os.path.join(data_dir, "binary_mask.pt"), weights_only=True
    )
    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    F_nodes, N = X.shape
    aki_idx = feature_names.index("aki_event")
    y_all_np = X[aki_idx].numpy()

    # ── Move full data to device ONCE ──
    X_full = X.to(device)
    y_all = torch.tensor(y_all_np, dtype=torch.float32, device=device)
    bm = binary_mask.to(device)

    print(f"  Data: {F_nodes} nodes × {N} patients")
    print(f"  AKI prevalence: {y_all_np.mean():.3f}")
    print(f"  Architecture: {config['arch']}")
    print(f"  Edge method: {config['edge_method']}, k={config['k']}")
    print(f"  in_channels (fixed): {N}")

    skf = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), y_all_np)):
        print(f"\n  ── Fold {fold+1}/{config['cv_folds']} ──")
        print(f"    Train: {len(train_idx)} patients, Test: {len(test_idx)} patients")

        # Build graph from training fold ONLY (no leakage)
        X_train_np = X[:, train_idx].cpu().numpy().T  # [N_train, F]
        if config["edge_method"] == "spearman":
            adj = build_spearman(X_train_np)
        elif config["edge_method"] == "mi":
            adj = build_mi(X_train_np)
        else:
            # LLM edges: precomputed, no data leakage issue
            adj = np.load(os.path.join(data_dir, f"adj_{config['edge_method']}.npy"))

        edge_index = knn_sparsify(adj, k=config["k"]).to(device)

        # Build model — in_channels = N (full cohort), same every fold
        model = build_model(
            arch=config["arch"],
            in_channels=N,
            hidden=config["hidden"],
            latent=config["latent"],
            heads=config.get("heads", 4),
            dropout=config.get("dropout", 0.2),
        ).to(device)

        metrics = train_one_fold(
            model,
            X_full,
            edge_index,
            aki_idx,
            y_all,
            train_idx,
            test_idx,
            bm,
            config,
            fold_idx=fold,
            wandb_run=wandb_run,
        )
        fold_metrics.append(metrics)

        print(
            f"    AUROC={metrics.get('auroc', float('nan')):.4f}  "
            f"AUPRC={metrics.get('auprc', float('nan')):.4f}  "
            f"Brier={metrics.get('brier', float('nan')):.4f}  "
            f"epoch={metrics.get('best_epoch', '?')}"
        )

    # Summary
    aurocs = [
        m["auroc"] for m in fold_metrics if not np.isnan(m.get("auroc", float("nan")))
    ]
    auprcs = [
        m["auprc"] for m in fold_metrics if not np.isnan(m.get("auprc", float("nan")))
    ]

    summary = {
        "auroc_mean": np.mean(aurocs) if aurocs else float("nan"),
        "auroc_std": np.std(aurocs) if aurocs else float("nan"),
        "auprc_mean": np.mean(auprcs) if auprcs else float("nan"),
        "auprc_std": np.std(auprcs) if auprcs else float("nan"),
        "folds": fold_metrics,
    }

    print(f"\n  ── Summary ──")
    print(f"    AUROC: {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")
    print(f"    AUPRC: {summary['auprc_mean']:.4f} ± {summary['auprc_std']:.4f}")

    if wandb_run is not None:
        wandb_run.summary["auroc_mean"] = summary["auroc_mean"]
        wandb_run.summary["auroc_std"] = summary["auroc_std"]
        wandb_run.summary["auprc_mean"] = summary["auprc_mean"]
        wandb_run.summary["auprc_std"] = summary["auprc_std"]

    return summary


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default="transformer",
        choices=["transformer", "gatv2", "gcn"],
    )
    parser.add_argument(
        "--edge_method", type=str, default="spearman", choices=["spearman", "mi", "llm"]
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="graphai-ici-aki")
    args = parser.parse_args()

    config = vars(args)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            config=config,
            group=f"{args.arch}_{args.edge_method}_k{args.k}",
            tags=["npj-pilot", args.arch, args.edge_method],
        )

    print("=" * 70)
    print("GRAPH AI — Training")
    print("=" * 70)

    summary = run_cv(config, data_dir=args.data_dir, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
