#!/usr/bin/env python3
"""
heatmap.py — Extract GATv2 attention weights → hierarchical clustering heatmap.

Trains the best model on full data, extracts layer-2 attention weights,
builds a [38×38] attention matrix, and plots a clustermap with seaborn.

Usage:
    python heatmap.py --edge_method spearman --run_name heatmap_sp
    python heatmap.py --edge_method llm --use_edge_weights --run_name heatmap_llm
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from graph import build_mi, build_spearman, knn_sparsify, load_llm_graph
from models import build_model


def extract_attention_matrix(model, x, edge_index, edge_attr, num_features):
    """Extract attention weights → dense [F, F] matrix."""
    attn_data = model.get_attention(x, edge_index, edge_attr=edge_attr)
    if attn_data is None:
        raise ValueError("Model does not support attention extraction")

    # Use layer 2 attention (final layer, closest to prediction)
    ei = attn_data["layer2"]["edge_index"]  # [2, E]
    aw = attn_data["layer2"]["attention"]  # [E, heads] or [E]

    # Average across heads if multi-head
    if aw.dim() > 1:
        aw = aw.mean(dim=1)

    # Build dense matrix
    A = torch.zeros(num_features, num_features)
    for idx in range(ei.size(1)):
        src, tgt = ei[0, idx].item(), ei[1, idx].item()
        A[tgt, src] = aw[idx].item()  # attention from src → tgt

    return A.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_method", default="spearman", choices=["spearman", "mi", "llm"]
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--use_edge_weights", action="store_true")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--run_name", default="heatmap")
    parser.add_argument("--output", default="figures/attention_heatmap.png")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load data
    X = torch.load(os.path.join(args.data_dir, "feature_matrix.pt"), weights_only=True)
    with open(os.path.join(args.data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    F_nodes, N_all = X.shape
    aki_idx = feature_names.index("aki_event")
    y_all = X[aki_idx].numpy()
    non_aki_idx = [i for i in range(F_nodes) if i != aki_idx]

    # 6-month landmark filter
    meta = pd.read_csv(os.path.join(args.data_dir, "cohort_meta.csv"))
    surv_days = meta["surv_days"].values
    eligible = (y_all == 1) | (surv_days >= 180)
    X = X[:, eligible]
    y_all = y_all[eligible]
    N = X.shape[1]
    print(f"  Data: {F_nodes} features × {N} patients (6mo landmark)")

    # Build graph
    if args.edge_method in ("spearman", "mi"):
        X_np = X.numpy().T  # [N, F]
        if args.edge_method == "spearman":
            adj = build_spearman(X_np)
        else:
            adj = build_mi(X_np)
        edge_index, edge_weight = knn_sparsify(adj, k=args.k, directed=False)
    else:
        edge_index, edge_weight = load_llm_graph(args.data_dir, k=args.k)

    edge_attr = edge_weight.unsqueeze(-1) if args.use_edge_weights else None
    edge_dim = 1 if args.use_edge_weights else None

    # Build and train model on full data
    model = build_model(
        num_features=F_nodes,
        aki_idx=aki_idx,
        hidden=args.hidden,
        latent=args.latent,
        heads=args.heads,
        dropout=args.dropout,
        edge_dim=edge_dim,
    )

    # Warm-start skip
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[non_aki_idx].numpy().T)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(X_scaled, y_all)
    with torch.no_grad():
        model.raw_skip.weight.copy_(torch.tensor(lr.coef_, dtype=torch.float32))
        model.raw_skip.bias.copy_(torch.tensor(lr.intercept_, dtype=torch.float32))

    # Prepare input
    x_input = X.clone()
    x_input[aki_idx, :] = 0.0
    # Scale
    x_np = x_input[non_aki_idx].numpy().T
    x_np = scaler.transform(x_np)
    x_input[non_aki_idx] = torch.tensor(x_np.T, dtype=torch.float32)

    # Train
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = torch.tensor((y_all == 0).sum() / (y_all == 1).sum()).clamp(max=10.0)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)

    print(f"  Training {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        logits, beta, z = model(x_input, edge_index, edge_attr=edge_attr)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_tensor,
            pos_weight=pos_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    # Extract attention
    print("  Extracting attention weights...")
    A = extract_attention_matrix(model, x_input, edge_index, edge_attr, F_nodes)

    # ── Plot heatmap with hierarchical clustering ─────────────
    os.makedirs(os.path.dirname(args.output) or "figures", exist_ok=True)

    # Clean feature names for display
    display_names = [n.replace("_", " ").title() for n in feature_names]

    # Create clustermap
    fig = sns.clustermap(
        pd.DataFrame(A, index=display_names, columns=display_names),
        cmap="YlOrRd",
        figsize=(12, 10),
        linewidths=0.3,
        linecolor="white",
        method="ward",
        metric="euclidean",
        annot=False,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Attention weight", "shrink": 0.6},
        dendrogram_ratio=(0.12, 0.12),
    )
    fig.ax_heatmap.set_xlabel("")
    fig.ax_heatmap.set_ylabel("")
    fig.ax_heatmap.tick_params(axis="x", labelsize=7, rotation=45)
    fig.ax_heatmap.tick_params(axis="y", labelsize=7)
    plt.suptitle(
        f"GATv2 Attention Heatmap ({args.edge_method.upper()}, k={args.k})",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"  Saved: {args.output}")

    # Also save raw attention matrix
    np.save(args.output.replace(".png", "_matrix.npy"), A)
    print(f"  Saved: {args.output.replace('.png', '_matrix.npy')}")

    # Print top attention edges
    print("\n  Top 15 attention edges:")
    edges = []
    for i in range(F_nodes):
        for j in range(F_nodes):
            if A[i, j] > 0 and i != j:
                edges.append((feature_names[j], feature_names[i], A[i, j]))
    edges.sort(key=lambda x: -x[2])
    for src, tgt, w in edges[:15]:
        print(f"    {src:30s} → {tgt:30s}  {w:.4f}")


if __name__ == "__main__":
    main()
