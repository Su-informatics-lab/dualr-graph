#!/usr/bin/env python3
"""
02_graph.py — Build feature-interaction graph.

Three edge construction methods:
  1. spearman  — |Spearman ρ| between feature vectors (handles binary+continuous)
  2. mi        — Normalized mutual information (discretizes continuous vars)
  3. llm       — LLM-estimated pairwise associations (DualR bridge)

Each method produces:
  - data/graph_{method}_k{k}.pt         edge_index tensor
  - data/adj_{method}.npy               raw adjacency matrix (for visualization)

Usage:
    python 02_graph.py --method spearman --k 8
    python 02_graph.py --method mi --k 8
    python 02_graph.py --method llm --k 8 --llm_prior data/llm_adj.npy

For LLM-prior edges, run 12_llm_edges.py first to generate data/llm_adj.npy.
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score

OUT = "data"


def build_spearman(X: np.ndarray) -> np.ndarray:
    """Spearman correlation adjacency. X: [N_patients, F_features]."""
    corr, _ = spearmanr(X)
    if corr.ndim == 0:
        # edge case: single feature
        return np.array([[0.0]])
    adj = np.abs(corr)
    np.fill_diagonal(adj, 0)
    return adj


def build_mi(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Normalized mutual information adjacency. X: [N, F]."""
    F = X.shape[1]
    adj = np.zeros((F, F))
    # Pre-discretize all columns
    X_disc = np.zeros_like(X, dtype=int)
    for i in range(F):
        col = X[:, i]
        unique = np.unique(col)
        if len(unique) <= 2:
            # Binary: keep as-is
            X_disc[:, i] = col.astype(int)
        else:
            bins = np.linspace(col.min() - 1e-8, col.max() + 1e-8, n_bins + 1)
            X_disc[:, i] = np.digitize(col, bins)

    for i in range(F):
        for j in range(i + 1, F):
            mi = normalized_mutual_info_score(X_disc[:, i], X_disc[:, j])
            adj[i, j] = adj[j, i] = mi
    return adj


def knn_sparsify(adj: np.ndarray, k: int, min_weight: float = 0.01) -> torch.Tensor:
    """k-NN sparsification → edge_index [2, E] with self-loops."""
    F = adj.shape[0]
    edges = set()
    for i in range(F):
        topk = np.argsort(adj[i])[-k:]
        for j in topk:
            if i != j and adj[i, j] > min_weight:
                edges.add((i, j))
                edges.add((j, i))  # undirected

    if not edges:
        # Fallback: connect all pairs above threshold
        for i in range(F):
            for j in range(F):
                if i != j and adj[i, j] > min_weight:
                    edges.add((i, j))

    edge_list = sorted(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Add self-loops
    self_loops = torch.arange(F, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    return edge_index


def print_graph_stats(
    adj: np.ndarray, edge_index: torch.Tensor, feature_names: list, method: str, k: int
):
    F = adj.shape[0]
    n_edges = edge_index.shape[1] - F  # subtract self-loops
    density = n_edges / (F * (F - 1)) if F > 1 else 0

    print(f"\n  Graph: {method}, k={k}")
    print(f"    Nodes: {F}")
    print(f"    Edges: {n_edges} (excl. self-loops)")
    print(f"    Density: {density:.3f}")

    # Degree distribution
    src = edge_index[0].numpy()
    degrees = np.bincount(src, minlength=F) - 1  # subtract self-loop
    print(
        f"    Degree: mean={degrees.mean():.1f}, "
        f"min={degrees.min()}, max={degrees.max()}"
    )

    # Top-3 strongest edges
    flat = adj.flatten()
    top_idx = np.argsort(flat)[-6:][::-1]  # top 6 (3 pairs, undirected)
    seen = set()
    print(f"    Top edges:")
    for idx in top_idx:
        i, j = divmod(idx, F)
        if (min(i, j), max(i, j)) in seen:
            continue
        seen.add((min(i, j), max(i, j)))
        print(
            f"      {feature_names[i]:25s} ↔ {feature_names[j]:25s}  "
            f"w={adj[i, j]:.3f}"
        )
        if len(seen) >= 3:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, default="spearman", choices=["spearman", "mi", "llm"]
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--min_weight", type=float, default=0.01)
    parser.add_argument("--data_dir", type=str, default=OUT)
    parser.add_argument(
        "--llm_prior",
        type=str,
        default=None,
        help="Path to precomputed LLM adjacency (data/llm_adj.npy)",
    )
    args = parser.parse_args()

    # Load feature matrix [F, N] and names
    X_tensor = torch.load(
        os.path.join(args.data_dir, "feature_matrix.pt"), weights_only=True
    )
    with open(os.path.join(args.data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    F, N = X_tensor.shape
    X = X_tensor.numpy().T  # [N, F] for correlation computation

    print("=" * 70)
    print(f"GRAPH AI — Build Feature Graph ({args.method})")
    print("=" * 70)
    print(f"  Features: {F}, Patients: {N}")

    # Build adjacency
    if args.method == "spearman":
        adj = build_spearman(X)
    elif args.method == "mi":
        adj = build_mi(X)
    elif args.method == "llm":
        if args.llm_prior is None:
            raise ValueError(
                "--llm_prior required for method=llm. " "Run 12_llm_edges.py first."
            )
        adj = np.load(args.llm_prior)
        if adj.shape != (F, F):
            raise ValueError(f"LLM adjacency shape {adj.shape} != ({F}, {F})")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Sparsify
    edge_index = knn_sparsify(adj, k=args.k, min_weight=args.min_weight)

    # Stats
    print_graph_stats(adj, edge_index, feature_names, args.method, args.k)

    # Save
    graph_path = os.path.join(args.data_dir, f"graph_{args.method}_k{args.k}.pt")
    adj_path = os.path.join(args.data_dir, f"adj_{args.method}.npy")
    torch.save(edge_index, graph_path)
    np.save(adj_path, adj)
    print(f"\n  Saved: {graph_path}")
    print(f"  Saved: {adj_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
