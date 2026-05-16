#!/usr/bin/env python3
"""
09_heatmap.py — Extract attention weights → feature interaction heatmap.

Loads a trained TransformerConv or GATv2 model and extracts the
encoder's first-layer attention weights as an F×F interaction matrix.

Output figures:
  - figures/heatmap_full.pdf         Full F×F attention heatmap
  - figures/heatmap_aki_row.pdf      AKI node → all features (bar chart)
  - figures/graph_topk.pdf           Top-k edges as network graph

Usage:
    python 09_heatmap.py --model_path results/best_model.pt \
        --arch transformer --edge_method spearman --k 8
"""

import argparse
import json
import os

import numpy as np
import torch

OUT = "figures"


def extract_heatmap(model, x, edge_index, feature_names):
    """Average attention across heads → F×F matrix."""
    model.eval()
    with torch.no_grad():
        ei, alpha = model.get_attention(x, edge_index)
        alpha_mean = alpha.mean(dim=-1)

    F = len(feature_names)
    heatmap = np.zeros((F, F))
    for k in range(ei.shape[1]):
        src, dst = ei[0, k].item(), ei[1, k].item()
        if src < F and dst < F:
            heatmap[src, dst] = alpha_mean[k].item()

    heatmap = (heatmap + heatmap.T) / 2
    return heatmap


def plot_full_heatmap(heatmap, feature_names, out_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Attention weight", "shrink": 0.8},
    )
    ax.set_title("Feature–Feature Interaction (Learned Attention)", fontsize=12)
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_aki_row(heatmap, feature_names, aki_idx, out_path):
    import matplotlib.pyplot as plt

    row = heatmap[aki_idx].copy()
    row[aki_idx] = 0  # remove self-attention

    order = np.argsort(row)[::-1]
    names = [feature_names[i] for i in order if i != aki_idx]
    values = [row[i] for i in order if i != aki_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c" if v > np.median(values) else "#3498db" for v in values]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Attention weight (from AKI node)", fontsize=10)
    ax.set_title(
        "Which features does the model attend to\nwhen reconstructing AKI?", fontsize=11
    )
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_graph_topk(heatmap, feature_names, out_path, top_k=30):
    """Network graph of top-k strongest edges."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("  SKIP graph plot (networkx not installed)")
        return

    F = len(feature_names)
    G = nx.Graph()
    for i in range(F):
        G.add_node(feature_names[i])

    # Collect all edges with weights
    edges = []
    for i in range(F):
        for j in range(i + 1, F):
            if heatmap[i, j] > 0:
                edges.append((feature_names[i], feature_names[j], heatmap[i, j]))
    edges.sort(key=lambda x: x[2], reverse=True)
    edges = edges[:top_k]

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42, k=2)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [3 * w / max_w for w in weights]

    nx.draw_networkx_nodes(
        G, pos, node_size=300, node_color="#3498db", alpha=0.8, ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, ax=ax)

    ax.set_title(f"Top-{top_k} Feature Interactions", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--arch", type=str, default="transformer", choices=["transformer", "gatv2"]
    )
    parser.add_argument("--edge_method", type=str, default="spearman")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=OUT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    X = torch.load(os.path.join(args.data_dir, "feature_matrix.pt"), weights_only=True)
    with open(os.path.join(args.data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    edge_index = torch.load(
        os.path.join(args.data_dir, f"graph_{args.edge_method}_k{args.k}.pt"),
        weights_only=True,
    )

    F_nodes, N = X.shape
    aki_idx = feature_names.index("aki_event")

    # Load model
    from importlib import import_module

    models_mod = import_module("03_models")
    model = models_mod.build_model(
        arch=args.arch,
        in_channels=N,
        hidden=args.hidden,
        latent=args.latent,
        heads=args.heads,
    )
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

    print("=" * 70)
    print("GRAPH AI — Heatmap Extraction")
    print("=" * 70)

    heatmap = extract_heatmap(model, X, edge_index, feature_names)

    # Save raw heatmap
    np.save(os.path.join(args.output_dir, "attention_heatmap.npy"), heatmap)

    # Plots
    plot_full_heatmap(
        heatmap, feature_names, os.path.join(args.output_dir, "heatmap_full.pdf")
    )
    plot_aki_row(
        heatmap,
        feature_names,
        aki_idx,
        os.path.join(args.output_dir, "heatmap_aki_row.pdf"),
    )
    plot_graph_topk(
        heatmap, feature_names, os.path.join(args.output_dir, "graph_topk.pdf")
    )

    print("Done.")


if __name__ == "__main__":
    main()
