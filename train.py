#!/usr/bin/env python3
"""
train.py — Training loop for Graph AI autoencoder.

Key design (per Dr. Su + Johnson et al. 2024):
  - Masked reconstruction: missing values filled by AE output each epoch,
    loss computed only on non-missing entries
  - Edge weights (LLM P(a|b)) passed to GATv2 as edge_attr
  - AKI node is masked for test patients (no data leakage)
  - Semi-supervised: all patients contribute to reconstruction,
    only labeled train patients contribute to CE loss
  - 6-month Cr≥1.5× primary endpoint

Usage:
    python train.py --arch gatv2 --edge_method spearman --k 8 --wandb
    python train.py --arch gatv2 --edge_method llm --k 8 --use_edge_weights --wandb
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from graph import build_mi, build_spearman, knn_sparsify, load_llm_graph
from models import build_model

# ══════════════════════════════════════════════════════════════
# Loss: reconstruction (non-missing only) + λ·CE (AKI node)
# ══════════════════════════════════════════════════════════════


def compute_loss(
    x_true,
    x_recon,
    aki_idx,
    y_aki,
    lam=1.0,
    binary_mask=None,
    missing_mask=None,
):
    """
    Semi-supervised loss with masked reconstruction.

    - Reconstruction loss: computed only on NON-MISSING entries,
      excluding the AKI row (which has its own CE loss).
    - CE loss: on AKI node for labeled (train) patients only.

    Args:
        x_true: [F, N] original values (NaN → 0 for missing)
        x_recon: [F, N] reconstructed
        aki_idx: int, index of AKI node
        y_aki: [N] ground truth (0/1, NaN for test patients)
        lam: CE weight
        binary_mask: [F] bool, True for binary features
        missing_mask: [F, N] bool, True where value was missing
    """
    F_nodes, N = x_true.shape

    # Build valid mask: non-missing, non-AKI entries
    if missing_mask is not None:
        valid = ~missing_mask.clone()
    else:
        valid = torch.ones_like(x_true, dtype=torch.bool)
    valid[aki_idx, :] = False  # AKI has separate CE loss

    if valid.sum() == 0:
        recon_loss = torch.tensor(0.0, device=x_true.device)
    elif binary_mask is not None:
        # Split binary vs continuous reconstruction loss
        cont_mask = ~binary_mask.clone()
        cont_mask[aki_idx] = False  # exclude AKI from continuous too
        bin_mask_2d = binary_mask.unsqueeze(1).expand_as(x_true) & valid
        cont_mask_2d = cont_mask.unsqueeze(1).expand_as(x_true) & valid

        loss_parts = []
        if bin_mask_2d.sum() > 0:
            loss_parts.append(
                F.binary_cross_entropy_with_logits(
                    x_recon[bin_mask_2d], x_true[bin_mask_2d]
                )
            )
        if cont_mask_2d.sum() > 0:
            loss_parts.append(F.mse_loss(x_recon[cont_mask_2d], x_true[cont_mask_2d]))
        recon_loss = sum(loss_parts) / max(len(loss_parts), 1)
    else:
        recon_loss = F.mse_loss(x_recon[valid], x_true[valid])

    # CE loss on AKI node (only labeled patients)
    aki_logits = x_recon[aki_idx]
    labeled = ~torch.isnan(y_aki)
    if labeled.sum() == 0:
        ce_loss = torch.tensor(0.0, device=x_true.device)
    else:
        ce_loss = F.binary_cross_entropy_with_logits(
            aki_logits[labeled], y_aki[labeled]
        )

    total = recon_loss + lam * ce_loss
    return total, recon_loss, ce_loss


# ══════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════


def evaluate(model, x_input, edge_index, aki_idx, y_test, test_idx, edge_attr=None):
    """AUROC + AUPRC on held-out patients (transductive: slice by test_idx)."""
    model.eval()
    with torch.no_grad():
        x_recon, _ = model(x_input, edge_index, edge_attr=edge_attr)
        probs = torch.sigmoid(x_recon[aki_idx, test_idx]).cpu().numpy()
    y = y_test.cpu().numpy()
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": roc_auc_score(y, probs),
        "auprc": average_precision_score(y, probs),
    }


# ══════════════════════════════════════════════════════════════
# Single fold training with masked reconstruction
# ══════════════════════════════════════════════════════════════


def train_one_fold(
    model,
    x_full,
    edge_index,
    aki_idx,
    train_idx,
    test_idx,
    binary_mask,
    missing_mask,
    config,
    edge_attr=None,
    fold_idx=0,
    wandb_run=None,
):
    """
    Train one fold with masked reconstruction for missing values.

    Key: x_full is the FULL [F, N] matrix. We don't split into separate
    train/test tensors because the graph autoencoder is TRANSDUCTIVE —
    all patients are in the forward pass. We control information flow via:
      - AKI row: test patients masked to 0
      - Missing values: filled with previous epoch's reconstruction
      - CE loss: only on train patients' AKI values
      - Recon loss: only on non-missing entries
    """
    device = x_full.device
    F_nodes, N = x_full.shape

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    # Prepare labels: train patients have labels, test patients are NaN
    y_all = torch.full((N,), float("nan"), device=device)
    y_all[train_idx] = x_full[aki_idx, train_idx].clone()

    # Ground truth for evaluation
    y_test = x_full[aki_idx, test_idx].clone()

    # Working copy of input — modified each epoch
    x_input = x_full.clone()
    # Mask test patients' AKI values
    x_input[aki_idx, test_idx] = 0.0

    best_auroc = 0.0
    best_metrics = {}
    patience_counter = 0
    best_state = None

    for epoch in range(config["epochs"]):
        model.train()
        x_recon, z = model(x_input, edge_index, edge_attr=edge_attr)

        loss, recon_l, ce_l = compute_loss(
            x_full,
            x_recon,
            aki_idx,
            y_all,
            lam=config["lambda_ce"],
            binary_mask=binary_mask,
            missing_mask=missing_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # ── Masked reconstruction: fill missing with AE output ──
        # (Dr. Su: "每一个epoch都用rec的value来替换missing，但不要替换non-missing value")
        with torch.no_grad():
            if missing_mask is not None:
                x_input[missing_mask] = x_recon[missing_mask].detach()
            # Keep test AKI masked
            x_input[aki_idx, test_idx] = 0.0

        # Evaluate periodically
        if epoch % config.get("eval_every", 10) == 0:
            metrics = evaluate(
                model,
                x_input,
                edge_index,
                aki_idx,
                y_test,
                test_idx,
                edge_attr=edge_attr,
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"fold{fold_idx}/loss": loss.item(),
                        f"fold{fold_idx}/recon": recon_l.item(),
                        f"fold{fold_idx}/ce": ce_l.item(),
                        f"fold{fold_idx}/auroc": metrics["auroc"],
                        f"fold{fold_idx}/auprc": metrics["auprc"],
                        f"fold{fold_idx}/epoch": epoch,
                    }
                )

            if (
                not np.isnan(metrics.get("auroc", float("nan")))
                and metrics["auroc"] > best_auroc
            ):
                best_auroc = metrics["auroc"]
                best_metrics = metrics.copy()
                best_metrics["best_epoch"] = epoch
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= config.get("patience", 30):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics


# ══════════════════════════════════════════════════════════════
# Cross-validation driver
# ══════════════════════════════════════════════════════════════


def run_cv(config, data_dir="data", wandb_run=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X = torch.load(os.path.join(data_dir, "feature_matrix.pt"), weights_only=True)
    binary_mask = torch.load(
        os.path.join(data_dir, "binary_mask.pt"), weights_only=True
    )
    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    F_nodes, N = X.shape
    aki_idx = feature_names.index("aki_event")
    y_all = X[aki_idx].numpy()

    # Load missing mask
    miss_path = os.path.join(data_dir, "missing_mask.pt")
    if os.path.exists(miss_path):
        missing_mask = torch.load(miss_path, weights_only=True).to(device)
        n_miss = missing_mask.sum().item()
        print(f"  Missing mask: {n_miss} entries")
    else:
        missing_mask = None
        print("  No missing mask found")

    # Edge weights
    use_ew = config.get("use_edge_weights", False)
    edge_dim = 1 if use_ew else None

    print(f"  Data: {F_nodes} nodes × {N} patients")
    print(f"  AKI prevalence: {y_all.mean():.3f}")
    print(f"  Architecture: {config['arch']}")
    print(f"  Edge method: {config['edge_method']}, k={config['k']}")
    print(f"  Edge weights: {'yes (edge_dim=1)' if use_ew else 'no'}")
    print(f"  Missing imputation: masked reconstruction")

    skf = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), y_all)):
        print(f"\n  ── Fold {fold+1}/{config['cv_folds']} ──")

        X_dev = X.to(device)

        # Build graph from TRAINING patients only (Spearman/MI)
        # LLM graph is precomputed and data-independent
        if config["edge_method"] in ("spearman", "mi"):
            X_train_np = X[:, train_idx].cpu().numpy().T  # [N_train, F]
            if config["edge_method"] == "spearman":
                adj = build_spearman(X_train_np)
            else:
                adj = build_mi(X_train_np)
            edge_index, edge_weight = knn_sparsify(adj, k=config["k"], directed=False)
        else:
            # LLM: load precomputed directed graph
            edge_index, edge_weight = load_llm_graph(
                data_dir,
                k=config["k"],
                cv_cutoff=config.get("cv_cutoff"),
            )

        edge_index = edge_index.to(device)
        edge_attr = edge_weight.unsqueeze(-1).to(device) if use_ew else None
        bm = binary_mask.to(device)
        mm = missing_mask  # already on device or None

        # Build model
        model = build_model(
            arch=config["arch"],
            in_channels=N,  # transductive: all patients in forward pass
            hidden=config["hidden"],
            latent=config["latent"],
            heads=config.get("heads", 4),
            dropout=config.get("dropout", 0.2),
            edge_dim=edge_dim,
        ).to(device)

        metrics = train_one_fold(
            model,
            X_dev,
            edge_index,
            aki_idx,
            train_idx,
            test_idx,
            bm,
            mm,
            config,
            edge_attr=edge_attr,
            fold_idx=fold,
            wandb_run=wandb_run,
        )
        fold_metrics.append(metrics)
        print(
            f"    AUROC={metrics.get('auroc', float('nan')):.4f} "
            f"AUPRC={metrics.get('auprc', float('nan')):.4f} "
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
    print(f"  AUROC: {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")
    print(f"  AUPRC: {summary['auprc_mean']:.4f} ± {summary['auprc_std']:.4f}")

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "auroc_mean": summary["auroc_mean"],
                "auroc_std": summary["auroc_std"],
                "auprc_mean": summary["auprc_mean"],
                "auprc_std": summary["auprc_std"],
            }
        )

    return summary


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", default="gatv2", choices=["gatv2", "gcn", "transformer"]
    )
    parser.add_argument(
        "--edge_method", default="spearman", choices=["spearman", "mi", "llm"]
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument(
        "--use_edge_weights",
        action="store_true",
        help="Pass edge weights as features to GATv2 attention",
    )
    parser.add_argument(
        "--cv_cutoff",
        type=float,
        default=None,
        help="LLM edge uncertainty cutoff (CV threshold)",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="graphai-ici-aki")
    args = parser.parse_args()

    config = vars(args)

    wandb_run = None
    if args.wandb:
        import wandb

        run_name = args.run_name or f"{args.arch}_{args.edge_method}_k{args.k}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            tags=["v6", args.arch, args.edge_method],
        )

    print("=" * 70)
    print("GRAPH AI — Training")
    print("=" * 70)

    summary = run_cv(config, data_dir=args.data_dir, wandb_run=wandb_run)

    # Save results
    os.makedirs("results", exist_ok=True)
    rname = args.run_name or f"{args.arch}_{args.edge_method}"
    out_path = f"results/{rname}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
