#!/usr/bin/env python3
"""
models_v4.py — Feature-graph autoencoder with association decoder.

Key change from v3: replaces per-patient value reconstruction with
feature-feature association reconstruction (z @ z.T / sqrt(d) → sigmoid).
This directly regularises embeddings to capture inter-feature structure.

AKI modes:
  pooled : global_mean_pool(z) → MLP  (default, matches reference)
  dual   : concat(z[aki_idx], pool(z)) → MLP  (GRAPE-style)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    TransformerConv,
    global_mean_pool,
)


@dataclass
class ModelOutput:
    z: torch.Tensor  # [B, F, emb]
    association_hat: torch.Tensor  # [B, F, F]
    aki_logits: torch.Tensor  # [B, n_classes]


# ── Encoder blocks ───────────────────────────────────────────


class GraphBlock(nn.Module):
    """Single GNN layer with LayerNorm and ELU."""

    def __init__(self, conv_type, in_dim, out_dim, heads, concat, dropout, edge_dim):
        super().__init__()
        self.conv_type = conv_type.lower()
        if self.conv_type == "gcn":
            self.conv = GCNConv(in_dim, out_dim, normalize=True)
            actual_out = out_dim
        elif self.conv_type == "gat":
            self.conv = GATConv(
                in_dim,
                out_dim,
                heads=heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                residual=True,
            )
            actual_out = out_dim * heads if concat else out_dim
        elif self.conv_type == "gatv2":
            self.conv = GATv2Conv(
                in_dim,
                out_dim,
                heads=heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                residual=True,
            )
            actual_out = out_dim * heads if concat else out_dim
        elif self.conv_type == "graph_transformer":
            self.conv = TransformerConv(
                in_dim,
                out_dim,
                heads=heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                beta=True,
            )
            actual_out = out_dim * heads if concat else out_dim
        else:
            raise ValueError(f"Unknown conv_type={self.conv_type!r}")
        self.norm = nn.LayerNorm(actual_out)
        self.dropout = float(dropout)
        self.actual_out = actual_out

    def forward(self, x, edge_index, edge_attr=None):
        if self.conv_type == "gcn":
            ew = edge_attr.squeeze(-1) if edge_attr is not None else None
            x = self.conv(x, edge_index, edge_weight=ew)
        else:
            x = self.conv(x, edge_index, edge_attr=edge_attr)
        return F.dropout(F.elu(self.norm(x)), p=self.dropout, training=self.training)


# ── Main model ───────────────────────────────────────────────


class FeatureGraphAutoencoderAKI(nn.Module):
    """
    Feature-graph autoencoder with association reconstruction + AKI head.

    Architecture:
        encoder:  stacked GNN → per-node embeddings z  [B*F, emb]
        decoder:  sigmoid(z @ z.T / sqrt(d))  →  association matrix  [B, F, F]
        AKI head: pool(z) → MLP → logits  [B, n_classes]
    """

    def __init__(
        self,
        n_features: int,
        node_input_dim: int = 2,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        n_layers: int = 2,
        encoder_type: str = "gatv2",
        n_heads: int = 4,
        dropout: float = 0.1,
        n_classes: int = 2,
        edge_dim: int | None = None,
        use_feature_identity: bool = True,
        aki_mode: str = "pooled",
        aki_idx: int | None = None,
    ):
        super().__init__()
        assert aki_mode in ("pooled", "dual"), f"aki_mode={aki_mode!r}"
        self.n_features = n_features
        self.aki_mode = aki_mode
        self.aki_idx = aki_idx

        # Feature identity embedding (shared across all patients)
        identity_dim = hidden_dim if use_feature_identity else 0
        self.use_feature_identity = use_feature_identity
        self.feature_embedding = (
            nn.Embedding(n_features, identity_dim) if use_feature_identity else None
        )

        # Input projection: [value, observed_mask(, +identity)] → hidden
        self.input_proj = nn.Linear(node_input_dim + identity_dim, hidden_dim)

        # Encoder: stacked GNN blocks
        blocks = []
        current = hidden_dim
        for i in range(n_layers):
            is_last = i == n_layers - 1
            od = embedding_dim if is_last else hidden_dim
            oh = 1 if is_last else n_heads
            cc = not is_last  # concat heads except last layer
            blocks.append(
                GraphBlock(encoder_type, current, od, oh, cc, dropout, edge_dim)
            )
            current = blocks[-1].actual_out
        self.encoder = nn.ModuleList(blocks)

        # AKI prediction head
        if aki_mode == "pooled":
            self.aki_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        elif aki_mode == "dual":
            assert aki_idx is not None, "dual mode requires aki_idx"
            self.aki_head = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

    def encode(self, x, edge_index, feature_id, edge_attr=None):
        """Encode node inputs → per-node embeddings."""
        if self.use_feature_identity:
            x = torch.cat([x, self.feature_embedding(feature_id)], dim=-1)
        h = F.relu(self.input_proj(x))
        for block in self.encoder:
            h = block(h, edge_index, edge_attr)
        return h

    def decode_association(self, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Inner-product association decoder: sigmoid(z @ z.T / sqrt(d))."""
        z_by_graph = z.view(batch_size, self.n_features, -1)
        scores = torch.matmul(z_by_graph, z_by_graph.transpose(1, 2))
        scores = scores / math.sqrt(z_by_graph.size(-1))
        return torch.sigmoid(scores)

    def forward(self, batch) -> ModelOutput:
        z = self.encode(batch.x, batch.edge_index, batch.feature_id, batch.edge_attr)

        # Association reconstruction
        association_hat = self.decode_association(z, batch.num_graphs)

        # AKI prediction
        if self.aki_mode == "pooled":
            pooled = global_mean_pool(z, batch.batch)
            logits = self.aki_head(pooled)
        elif self.aki_mode == "dual":
            z_by_graph = z.view(batch.num_graphs, self.n_features, -1)
            z_aki = z_by_graph[:, self.aki_idx, :]
            z_pool = z_by_graph.mean(dim=1)
            logits = self.aki_head(torch.cat([z_aki, z_pool], dim=-1))

        return ModelOutput(z=z, association_hat=association_hat, aki_logits=logits)


# ── Loss functions ───────────────────────────────────────────


def association_mse_loss(
    association_hat: torch.Tensor,
    association_target: torch.Tensor,
    ignore_diagonal: bool = True,
) -> torch.Tensor:
    """MSE between predicted and target association matrices."""
    target = association_target.to(association_hat.device)
    target = target.unsqueeze(0).expand_as(association_hat)
    if not ignore_diagonal:
        return F.mse_loss(association_hat, target)
    n = target.size(-1)
    mask = ~torch.eye(n, dtype=torch.bool, device=target.device)
    return F.mse_loss(association_hat[:, mask], target[:, mask])


# ── Builder ──────────────────────────────────────────────────


def build_model(
    n_features: int,
    aki_idx: int,
    hidden_dim: int = 64,
    embedding_dim: int = 32,
    n_layers: int = 2,
    encoder_type: str = "gatv2",
    n_heads: int = 4,
    dropout: float = 0.1,
    n_classes: int = 2,
    edge_dim: int | None = None,
    aki_mode: str = "pooled",
    **_,
) -> FeatureGraphAutoencoderAKI:
    return FeatureGraphAutoencoderAKI(
        n_features=n_features,
        node_input_dim=2,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        encoder_type=encoder_type,
        n_heads=n_heads,
        dropout=dropout,
        n_classes=n_classes,
        edge_dim=edge_dim,
        aki_mode=aki_mode,
        aki_idx=aki_idx,
    )
