#!/usr/bin/env python3
"""
models.py — Graph Autoencoder architectures for feature-interaction graph.

Architecture: Encoder (2× GNN layers) → latent [F, d] → Decoder (linear) → [F, N]

Key design choices (per Johnson et al. 2024, Annu Rev Biomed Data Sci):
  - GATv2 as primary: learnable attention = ante hoc explainability
  - Edge weights (LLM P(a|b)) passed as edge_attr to bias attention
  - Shallow (2 layers) to avoid oversmoothing on 38-node graph
  - Attention weights extractable for heatmap visualization

Usage:
    model = build_model("gatv2", in_channels=2292, hidden=64, latent=16,
                        heads=4, dropout=0.2, edge_dim=1)
    x_recon, z = model(x, edge_index, edge_attr=edge_weight)
    attn = model.get_attention(x, edge_index, edge_attr=edge_weight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, TransformerConv


class GraphAE(nn.Module):
    """Graph Autoencoder base class."""

    def __init__(self, in_channels, hidden, latent, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.latent = latent
        self.dropout = dropout
        # Shared decoder for reconstruction (non-AKI nodes)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_channels),
        )
        # Dedicated AKI classifier — separate weights so AKI prediction
        # doesn't compete with heterogeneous feature reconstruction
        self.aki_head = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_channels),
        )

    def encode(self, x, edge_index, edge_attr=None):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr=None, aki_idx=None):
        z = self.encode(x, edge_index, edge_attr)
        x_recon = self.decoder(z)
        if aki_idx is not None:
            x_recon = x_recon.clone()
            x_recon[aki_idx] = self.aki_head(z[aki_idx])
        return x_recon, z

    def get_attention(self, x, edge_index, edge_attr=None):
        """Override in attention-based models to return attention weights."""
        return None


class GATv2AE(GraphAE):
    """
    GATv2 Autoencoder — primary architecture.

    Supports edge_attr (LLM-estimated P(a|b)) as attention bias.
    GATv2Conv uses dynamic attention: a^T LeakyReLU(W[x_i || x_j] + W_e·e_ij)
    When edge_dim is set, edge attributes are projected and added to the
    attention computation, biasing message passing by clinical knowledge.
    """

    def __init__(
        self, in_channels, hidden, latent, heads=4, dropout=0.2, edge_dim=None
    ):
        super().__init__(in_channels, hidden, latent, dropout)
        self.conv1 = GATv2Conv(
            in_channels,
            hidden,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=edge_dim,
        )
        self.conv2 = GATv2Conv(
            hidden * heads,
            latent,
            heads=1,
            dropout=dropout,
            concat=False,
            edge_dim=edge_dim,
        )
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.edge_dim = edge_dim

    def encode(self, x, edge_index, edge_attr=None):
        ea = edge_attr if self.edge_dim else None
        h = self.conv1(x, edge_index, edge_attr=ea)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index, edge_attr=ea)
        return z

    def get_attention(self, x, edge_index, edge_attr=None):
        """Extract attention weights from both layers."""
        self.eval()
        ea = edge_attr if self.edge_dim else None
        with torch.no_grad():
            # Layer 1
            h1, (ei1, aw1) = self.conv1(
                x, edge_index, edge_attr=ea, return_attention_weights=True
            )
            h1 = self.norm1(h1)
            h1 = F.elu(h1)
            # Layer 2
            _, (ei2, aw2) = self.conv2(
                h1, edge_index, edge_attr=ea, return_attention_weights=True
            )
        return {
            "layer1": {"edge_index": ei1.cpu(), "attention": aw1.cpu()},
            "layer2": {"edge_index": ei2.cpu(), "attention": aw2.cpu()},
        }


class GCNAE(GraphAE):
    """GCN Autoencoder — baseline without attention (undirected only)."""

    def __init__(self, in_channels, hidden, latent, dropout=0.2, **kwargs):
        super().__init__(in_channels, hidden, latent, dropout)
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, latent)
        self.norm1 = nn.LayerNorm(hidden)

    def encode(self, x, edge_index, edge_attr=None):
        # GCN can use edge_weight (scalar) but not edge_attr (vector)
        ew = (
            edge_attr.squeeze(-1)
            if edge_attr is not None and edge_attr.dim() > 1
            else edge_attr
        )
        h = self.conv1(x, edge_index, edge_weight=ew)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index, edge_weight=ew)
        return z


class TransformerAE(GraphAE):
    """TransformerConv Autoencoder — highest expressivity but may collapse."""

    def __init__(
        self, in_channels, hidden, latent, heads=4, dropout=0.2, edge_dim=None
    ):
        super().__init__(in_channels, hidden, latent, dropout)
        self.conv1 = TransformerConv(
            in_channels,
            hidden,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=edge_dim,
        )
        self.conv2 = TransformerConv(
            hidden * heads,
            latent,
            heads=1,
            dropout=dropout,
            concat=False,
            edge_dim=edge_dim,
        )
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.edge_dim = edge_dim

    def encode(self, x, edge_index, edge_attr=None):
        ea = edge_attr if self.edge_dim else None
        h = self.conv1(x, edge_index, edge_attr=ea)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index, edge_attr=ea)
        return z

    def get_attention(self, x, edge_index, edge_attr=None):
        self.eval()
        ea = edge_attr if self.edge_dim else None
        with torch.no_grad():
            _, aw1 = self.conv1(
                x, edge_index, edge_attr=ea, return_attention_weights=True
            )
            h = self.norm1(self.conv1(x, edge_index, edge_attr=ea))
            h = F.elu(h)
            _, aw2 = self.conv2(
                h, edge_index, edge_attr=ea, return_attention_weights=True
            )
        return {
            "layer1": {"edge_index": aw1[0].cpu(), "attention": aw1[1].cpu()},
            "layer2": {"edge_index": aw2[0].cpu(), "attention": aw2[1].cpu()},
        }


def build_model(
    arch: str = "gatv2",
    in_channels: int = 2292,
    hidden: int = 64,
    latent: int = 16,
    heads: int = 4,
    dropout: float = 0.2,
    edge_dim: int = None,
) -> GraphAE:
    """Factory function for graph autoencoder models."""
    if arch == "gatv2":
        return GATv2AE(in_channels, hidden, latent, heads, dropout, edge_dim)
    elif arch == "gcn":
        return GCNAE(in_channels, hidden, latent, dropout)
    elif arch == "transformer":
        return TransformerAE(in_channels, hidden, latent, heads, dropout, edge_dim)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
