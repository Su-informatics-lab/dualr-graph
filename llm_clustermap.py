#!/usr/bin/env python3
"""
llm_clustermap.py — Visualize raw LLM-estimated P(a|b) adjacency matrix
as a hierarchical clustering heatmap.

This shows the LLM's clinical knowledge priors BEFORE any GNN training.

Usage:
    python llm_clustermap.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_dir = "data"
adj = np.load(os.path.join(data_dir, "llm_adj.npy"))
with open(os.path.join(data_dir, "feature_names.json")) as f:
    names = json.load(f)

F = adj.shape[0]
print(f"LLM adjacency: {F}×{F}, directed (asymmetric)")
print(f"Non-zero entries: {np.count_nonzero(adj)}")
print(f"Value range: [{adj.min():.3f}, {adj.max():.3f}]")
print(f"AKI column (sink): all zeros = {(adj[:, names.index('aki_event')] == 0).all()}")

# Clean feature names
display = [n.replace("_", " ").title() for n in names]

# Symmetrize for clustering: average P(a|b) and P(b|a)
adj_sym = (adj + adj.T) / 2
np.fill_diagonal(adj_sym, 0)

df = pd.DataFrame(adj_sym, index=display, columns=display)

os.makedirs("figures", exist_ok=True)

# ── Clustermap ────────────────────────────────────────────────
g = sns.clustermap(
    df,
    cmap="YlOrRd",
    figsize=(13, 11),
    linewidths=0.2,
    linecolor="white",
    method="ward",
    metric="euclidean",
    vmin=0,
    vmax=0.7,
    annot=False,
    xticklabels=True,
    yticklabels=True,
    cbar_kws={"label": "Mean P(a|b), P(b|a)", "shrink": 0.5},
    dendrogram_ratio=(0.12, 0.12),
)
g.ax_heatmap.set_xlabel("")
g.ax_heatmap.set_ylabel("")
g.ax_heatmap.tick_params(axis="x", labelsize=7, rotation=45)
g.ax_heatmap.tick_params(axis="y", labelsize=7)

plt.suptitle(
    "LLM-estimated clinical feature associations\n"
    "Directed P(a|b) from Llama3.1-8b-instruct (10 reps, symmetrized for clustering)",
    y=1.02,
    fontsize=11,
    fontweight="bold",
)

out = "figures/llm_prob_clustermap.png"
g.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out}")

# ── Also print top associations ──────────────────────────────
print("\nTop 20 LLM associations (symmetrized):")
pairs = []
for i in range(F):
    for j in range(i + 1, F):
        if adj_sym[i, j] > 0:
            pairs.append((names[i], names[j], adj_sym[i, j], adj[i, j], adj[j, i]))
pairs.sort(key=lambda x: -x[2])
print(
    f"  {'Feature A':<30s}  {'Feature B':<30s}  {'Mean':>6s}  {'P(A|B)':>7s}  {'P(B|A)':>7s}"
)
for a, b, m, pab, pba in pairs[:20]:
    print(f"  {a:<30s}  {b:<30s}  {m:.3f}   {pab:.3f}    {pba:.3f}")
