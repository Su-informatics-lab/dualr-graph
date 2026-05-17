"""
03_models.py — Graph Autoencoder architectures.

Three variants sharing the same encoder-decoder skeleton:
  A. TransformerConv (primary)
  B. GATv2Conv (comparison)
  C. GCNConv (baseline)

All produce:
  - x_recon [F, N]: reconstructed feature matrix
  - z [F, d_latent]: latent embeddings
  - attention weights (A and B only)
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, TransformerConv


class GraphAE_Transformer(nn.Module):
    """Primary: TransformerConv encoder + decoder."""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        latent: int,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.enc1 = TransformerConv(
            in_channels, hidden, heads=heads, concat=False, dropout=dropout
        )
        self.enc2 = TransformerConv(
            hidden, latent, heads=heads, concat=False, dropout=dropout
        )
        self.dec1 = TransformerConv(
            latent, hidden, heads=heads, concat=False, dropout=dropout
        )
        self.dec2 = nn.Linear(hidden, in_channels)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        h = F.elu(self.enc1(x, edge_index))
        h = self.dropout(h)
        return F.elu(self.enc2(h, edge_index))

    def decode(self, z, edge_index):
        h = F.elu(self.dec1(z, edge_index))
        h = self.dropout(h)
        return self.dec2(h)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index), z

    def get_attention(self, x, edge_index):
        _, (ei, alpha) = self.enc1(x, edge_index, return_attention_weights=True)
        return ei, alpha  # alpha: [num_edges, heads]


class GraphAE_GATv2(nn.Module):
    """Comparison: GATv2Conv encoder + decoder."""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        latent: int,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.enc1 = GATv2Conv(
            in_channels, hidden, heads=heads, concat=False, dropout=dropout
        )
        self.enc2 = GATv2Conv(
            hidden, latent, heads=heads, concat=False, dropout=dropout
        )
        self.dec1 = GATv2Conv(
            latent, hidden, heads=heads, concat=False, dropout=dropout
        )
        self.dec2 = nn.Linear(hidden, in_channels)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        h = F.elu(self.enc1(x, edge_index))
        h = self.dropout(h)
        return F.elu(self.enc2(h, edge_index))

    def decode(self, z, edge_index):
        h = F.elu(self.dec1(z, edge_index))
        h = self.dropout(h)
        return self.dec2(h)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index), z

    def get_attention(self, x, edge_index):
        _, (ei, alpha) = self.enc1(x, edge_index, return_attention_weights=True)
        return ei, alpha


class GraphAE_GCN(nn.Module):
    """Baseline: GCNConv encoder + decoder. No attention weights."""

    def __init__(
        self, in_channels: int, hidden: int, latent: int, dropout: float = 0.2, **kwargs
    ):
        super().__init__()
        self.enc1 = GCNConv(in_channels, hidden)
        self.enc2 = GCNConv(hidden, latent)
        self.dec1 = GCNConv(latent, hidden)
        self.dec2 = nn.Linear(hidden, in_channels)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        h = F.elu(self.enc1(x, edge_index))
        h = self.dropout(h)
        return F.elu(self.enc2(h, edge_index))

    def decode(self, z, edge_index):
        h = F.elu(self.dec1(z, edge_index))
        h = self.dropout(h)
        return self.dec2(h)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index), z

    def get_attention(self, x, edge_index):
        raise NotImplementedError("GCN has no attention weights")


ARCH_REGISTRY = {
    "transformer": GraphAE_Transformer,
    "gatv2": GraphAE_GATv2,
    "gcn": GraphAE_GCN,
}


def build_model(
    arch: str,
    in_channels: int,
    hidden: int,
    latent: int,
    heads: int = 4,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory function."""
    cls = ARCH_REGISTRY[arch]
    if arch == "gcn":
        return cls(in_channels, hidden, latent, dropout=dropout)
    return cls(in_channels, hidden, latent, heads=heads, dropout=dropout)
