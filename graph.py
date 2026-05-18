#!/usr/bin/env python3
"""
graph.py — Feature-interaction graph construction.

Three edge methods:
  1. spearman  — |Spearman ρ| (undirected, data-derived)
  2. mi        — Normalized mutual information (undirected, data-derived)
  3. llm       — LLM-estimated P(a|b) (DIRECTED, knowledge-derived)

Output per method:
  data/graph_{method}_k{k}.pt   — dict with edge_index + edge_weight tensors
  data/adj_{method}.npy         — raw adjacency (for visualization)

Usage:
    python graph.py --method spearman --k 8
    python graph.py --method llm --k 8 --llm_prior data/llm_adj.npy
    python graph.py --method llm --k 8 --llm_prior data/llm_adj.npy --cv_cutoff 0.3
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score


def build_spearman(X: np.ndarray) -> np.ndarray:
    """Spearman correlation adjacency. X: [N_patients, F_features]."""
    F = X.shape[1]
    rho, _ = spearmanr(X)
    if rho.ndim == 0:
        rho = np.array([[1.0]])
    adj = np.abs(rho)
    np.fill_diagonal(adj, 0.0)
    return adj


def build_mi(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Normalized mutual information adjacency. X: [N_patients, F_features]."""
    F = X.shape[1]
    # Discretize continuous features for MI
    X_disc = np.zeros_like(X, dtype=int)
    for j in range(F):
        col = X[:, j]
        if len(np.unique(col)) <= n_bins:
            X_disc[:, j] = col.astype(int)
        else:
            X_disc[:, j] = np.digitize(
                col, np.linspace(col.min(), col.max(), n_bins + 1)[1:-1]
            )

    adj = np.zeros((F, F), dtype=np.float64)
    for i in range(F):
        for j in range(i + 1, F):
            nmi = normalized_mutual_info_score(X_disc[:, i], X_disc[:, j])
            adj[i, j] = nmi
            adj[j, i] = nmi
    return adj


def knn_sparsify(
    adj: np.ndarray,
    k: int = 8,
    directed: bool = False,
) -> tuple:
    """
    k-NN sparsification → (edge_index [2, E], edge_weight [E]).

    For undirected graphs (Spearman/MI): symmetrize by keeping edge if
    EITHER endpoint has the other in its top-k.

    For directed graphs (LLM): each node keeps its top-k INCOMING edges
    (top-k strongest predictors for each target node). adj[i,j] = P(i|j),
    so row i contains incoming weights for node i.
    """
    F = adj.shape[0]
    sources, targets, weights = [], [], []

    if directed:
        # For each target node i, keep top-k source nodes j by adj[i,j]
        for i in range(F):
            row = adj[i, :]
            nonzero = np.where(row > 0)[0]
            if len(nonzero) == 0:
                continue
            topk = nonzero[np.argsort(row[nonzero])[-min(k, len(nonzero)) :]]
            for j in topk:
                sources.append(int(j))  # edge from j → i
                targets.append(int(i))
                weights.append(float(adj[i, j]))
    else:
        # Symmetric: keep if either direction is in top-k
        mask = np.zeros_like(adj, dtype=bool)
        for i in range(F):
            row = adj[i, :]
            topk = np.argsort(row)[-min(k, F - 1) :]
            mask[i, topk] = True
        # Symmetrize
        mask = mask | mask.T
        for i in range(F):
            for j in range(F):
                if mask[i, j] and adj[i, j] > 0:
                    sources.append(i)
                    targets.append(j)
                    weights.append(float(adj[i, j]))

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def load_llm_graph(
    data_dir: str = "data",
    k: int = 8,
    cv_cutoff: float = None,
    llm_prior_path: str = None,
) -> tuple:
    """
    Load LLM-estimated directed adjacency → (edge_index, edge_weight).

    If cv_cutoff is set, edges with coefficient of variation > cv_cutoff
    are zeroed before sparsification (uncertainty-based pruning).
    """
    adj_path = llm_prior_path or os.path.join(data_dir, "llm_adj.npy")
    adj = np.load(adj_path)

    # Optional: uncertainty-based cutoff
    std_path = os.path.join(data_dir, "llm_adj_std.npy")
    if cv_cutoff is not None and os.path.exists(std_path):
        adj_std = np.load(std_path)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(adj > 0, adj_std / adj, np.inf)
        n_before = np.count_nonzero(adj)
        adj[cv > cv_cutoff] = 0.0
        n_after = np.count_nonzero(adj)
        print(f"  CV cutoff {cv_cutoff}: {n_before} → {n_after} edges")

    return knn_sparsify(adj, k=k, directed=True)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default="spearman", choices=["spearman", "mi", "llm"]
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--llm_prior", type=str, default=None)
    parser.add_argument(
        "--cv_cutoff",
        type=float,
        default=None,
        help="Remove LLM edges with CV > this threshold",
    )
    args = parser.parse_args()

    X = torch.load(
        os.path.join(args.data_dir, "feature_matrix.pt"), weights_only=True
    ).numpy()
    with open(os.path.join(args.data_dir, "feature_names.json")) as f:
        names = json.load(f)
    F, N = X.shape
    print(f"  Features: {F}, Patients: {N}")

    if args.method == "llm":
        edge_index, edge_weight = load_llm_graph(
            args.data_dir,
            k=args.k,
            cv_cutoff=args.cv_cutoff,
            llm_prior_path=args.llm_prior,
        )
        adj = np.load(args.llm_prior or os.path.join(args.data_dir, "llm_adj.npy"))
        directed = True
    else:
        X_T = X.T  # [N, F]
        if args.method == "spearman":
            adj = build_spearman(X_T)
        else:
            adj = build_mi(X_T)
        edge_index, edge_weight = knn_sparsify(adj, k=args.k, directed=False)
        directed = False

    tag = f"{'directed' if directed else 'undirected'}"
    print(f"  Method: {args.method} ({tag}), k={args.k}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Weight range: [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")

    # Save
    out = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "method": args.method,
        "k": args.k,
        "directed": directed,
    }
    graph_path = os.path.join(args.data_dir, f"graph_{args.method}_k{args.k}.pt")
    torch.save(out, graph_path)
    np.save(os.path.join(args.data_dir, f"adj_{args.method}.npy"), adj)
    print(f"  Saved: {graph_path}")


if __name__ == "__main__":
    main()
