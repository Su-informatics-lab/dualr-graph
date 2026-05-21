#!/usr/bin/env python3
"""
models_v3.py — Feature-graph autoencoder for ICI→AKI prediction.

v10.2 changes:
  ① ContextualValueDecoder: decoder(z_j, global_pool(z)) → x̂_j
     Pool types: mean, max, mean_max (concat both).
  ② GATv2 encoder option alongside DirectedWeightedConv.
  ③ Simplified loss: MSE on all 38 features + CE on AKI.
  ④ Iterative forward for refinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, MessagePassing


@dataclass
class ModelOutput:
    z: torch.Tensor
    x_hat: torch.Tensor
    aki_prob: torch.Tensor
    association_hat: torch.Tensor | None = None


# ─────────────────────────────────────────────────────────────
# Directed weighted message passing (Su's equation)
# ─────────────────────────────────────────────────────────────


class DirectedWeightedConv(MessagePassing):
    """v_i^(l+1) = W_self·v_i^(l) + Σ W_neigh·v_j^(l)·P(v_i|v_j)"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        return self.W_self(x) + self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight=None):
        msg = self.W_neigh(x_j)
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)
        return msg


# ─────────────────────────────────────────────────────────────
# Encoder blocks
# ─────────────────────────────────────────────────────────────


class DirectedGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.conv = DirectedWeightedConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout
        self.conv_type = "directed"

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv(x, edge_index, edge_weight)
        return F.dropout(F.elu(self.norm(h)), p=self.dropout, training=self.training)


class GATv2Block(nn.Module):
    def __init__(
        self, in_dim, out_dim, heads=4, dropout=0.1, edge_dim=None, is_last=False
    ):
        super().__init__()
        head_out = max(1, out_dim // heads)
        self.conv = GATv2Conv(
            in_dim,
            head_out,
            heads=heads,
            concat=not is_last,
            dropout=dropout,
            edge_dim=edge_dim,
            residual=True,
        )
        actual_out = head_out if is_last else head_out * heads
        self.proj = (
            nn.Linear(actual_out, out_dim) if actual_out != out_dim else nn.Identity()
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout
        self.conv_type = "gatv2"

    def forward(self, x, edge_index, edge_weight=None):
        edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = self.proj(h)
        return F.dropout(F.elu(self.norm(h)), p=self.dropout, training=self.training)


# ─────────────────────────────────────────────────────────────
# Contextual value decoder
# ─────────────────────────────────────────────────────────────


class ContextualValueDecoder(nn.Module):
    """
    decoder(z_j, global_context) → x̂_j

    Each node sees its own embedding PLUS a global summary of the patient.
    pool_type: 'mean', 'max', 'mean_max' (concat both).
    """

    POOL_TYPES = ("mean", "max", "mean_max")

    def __init__(self, embedding_dim: int, hidden_dim: int, pool_type: str = "mean"):
        super().__init__()
        assert pool_type in self.POOL_TYPES
        self.pool_type = pool_type
        ctx_dim = embedding_dim if pool_type in ("mean", "max") else 2 * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, z: torch.Tensor, batch_size: int, n_features: int
    ) -> torch.Tensor:
        z_bg = z.view(batch_size, n_features, -1)  # [B, F, emb]

        if self.pool_type == "mean":
            ctx = z_bg.mean(dim=1)  # [B, emb]
        elif self.pool_type == "max":
            ctx = z_bg.max(dim=1).values
        else:
            ctx = torch.cat(
                [z_bg.mean(dim=1), z_bg.max(dim=1).values], dim=-1
            )  # [B, 2*emb]

        ctx_exp = ctx.unsqueeze(1).expand(-1, n_features, -1)  # [B, F, ctx]
        combined = torch.cat([z_bg, ctx_exp], dim=-1)  # [B, F, emb+ctx]
        return self.mlp(combined.reshape(-1, combined.size(-1))).view(
            batch_size, n_features
        )


# ─────────────────────────────────────────────────────────────
# Association decoder (optional auxiliary)
# ─────────────────────────────────────────────────────────────


class SymmetricAssociationDecoder(nn.Module):
    def forward(self, z_by_graph):
        d = z_by_graph.size(-1)
        return torch.sigmoid(z_by_graph @ z_by_graph.transpose(1, 2) / math.sqrt(d))


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────


class FeatureGraphAutoencoderV3(nn.Module):

    def __init__(
        self,
        n_features: int,
        aki_idx: int,
        hidden_dim: int = 64,
        embedding_dim: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
        encoder_type: str = "directed",
        pool_type: str = "mean",
        n_heads: int = 4,
        use_assoc_aux: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.aki_idx = aki_idx
        self.embedding_dim = embedding_dim
        self.use_assoc_aux = use_assoc_aux

        # Input projection
        self.feature_embedding = nn.Embedding(n_features, hidden_dim)
        self.input_proj = nn.Linear(2 + hidden_dim, hidden_dim)

        # Encoder
        blocks = []
        for i in range(n_layers):
            in_d = hidden_dim
            out_d = embedding_dim if i == n_layers - 1 else hidden_dim
            if encoder_type == "directed":
                blocks.append(DirectedGCNBlock(in_d, out_d, dropout))
            elif encoder_type == "gatv2":
                blocks.append(
                    GATv2Block(
                        in_d,
                        out_d,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=1,
                        is_last=(i == n_layers - 1),
                    )
                )
            else:
                raise ValueError(f"Unknown encoder_type={encoder_type}")
        self.encoder = nn.ModuleList(blocks)

        # Contextual decoder
        self.value_decoder = ContextualValueDecoder(
            embedding_dim, hidden_dim, pool_type
        )

        # Optional association auxiliary
        if use_assoc_aux:
            self.assoc_decoder = SymmetricAssociationDecoder()

    def encode(self, x, edge_index, feature_id, edge_weight=None):
        identity = self.feature_embedding(feature_id)
        h = F.relu(self.input_proj(torch.cat([x, identity], dim=-1)))
        for block in self.encoder:
            h = block(h, edge_index, edge_weight)
        return h

    def decode(self, z, batch_size):
        return self.value_decoder(z, batch_size, self.n_features)

    def forward(self, batch, n_iter: int = 1) -> ModelOutput:
        edge_weight = getattr(batch, "edge_weight", None)
        x_current = batch.x

        for iteration in range(n_iter):
            z = self.encode(x_current, batch.edge_index, batch.feature_id, edge_weight)
            x_hat = self.decode(z, batch.num_graphs)
            if iteration < n_iter - 1:
                x_filled = x_current.clone()
                aki_pos = (batch.feature_id == self.aki_idx).nonzero(as_tuple=True)[0]
                x_filled[aki_pos, 0] = torch.sigmoid(x_hat[:, self.aki_idx]).detach()
                x_filled[aki_pos, 1] = 1.0
                x_current = x_filled

        aki_prob = torch.sigmoid(x_hat[:, self.aki_idx])

        assoc = None
        if self.use_assoc_aux:
            z_bg = z.view(batch.num_graphs, self.n_features, -1)
            assoc = self.assoc_decoder(z_bg)

        return ModelOutput(z=z, x_hat=x_hat, aki_prob=aki_prob, association_hat=assoc)


# ─────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────


def reconstruction_mse(x_hat, x_true, obs_mask):
    """MSE over all 38 features including AKI. Only observed entries."""
    diff_sq = (x_hat - x_true).pow(2)
    masked = diff_sq * obs_mask
    return masked.sum() / obs_mask.sum().clamp(min=1)


def aki_ce_loss(x_hat_aki, aki_true):
    """CE on AKI as additional emphasis. Logits for numerical stability."""
    return F.binary_cross_entropy_with_logits(
        x_hat_aki, aki_true.float(), reduction="mean"
    )


def association_aux_loss(association_hat, association_target, ignore_diag=True):
    if association_hat is None:
        return torch.tensor(0.0, device=association_target.device, requires_grad=True)
    target = association_target.unsqueeze(0).expand_as(association_hat)
    n_f = target.size(-1)
    if ignore_diag:
        mask = ~torch.eye(n_f, dtype=torch.bool, device=target.device)
        return F.mse_loss(association_hat[:, mask], target[:, mask])
    return F.mse_loss(association_hat, target)


def total_loss(
    out,
    x_true,
    obs_mask,
    aki_true,
    aki_idx,
    assoc_target=None,
    lambda_rec=1.0,
    lambda_aki=2.0,
    lambda_assoc=0.0,
):
    l_rec = reconstruction_mse(out.x_hat, x_true, obs_mask)
    l_aki = aki_ce_loss(out.x_hat[:, aki_idx], aki_true)
    l_assoc = torch.tensor(0.0, device=out.x_hat.device)
    if assoc_target is not None and out.association_hat is not None:
        l_assoc = association_aux_loss(out.association_hat, assoc_target)
    l_total = lambda_rec * l_rec + lambda_aki * l_aki + lambda_assoc * l_assoc
    return {
        "loss": l_total,
        "rec_loss": l_rec,
        "aki_loss": l_aki,
        "assoc_loss": l_assoc,
    }


# ─────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────


def build_model_v3(
    n_features: int,
    aki_idx: int,
    hidden_dim: int = 64,
    embedding_dim: int = 16,
    n_layers: int = 2,
    dropout: float = 0.0,
    encoder_type: str = "directed",
    pool_type: str = "mean",
    n_heads: int = 4,
    use_assoc_aux: bool = False,
) -> FeatureGraphAutoencoderV3:
    return FeatureGraphAutoencoderV3(
        n_features=n_features,
        aki_idx=aki_idx,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        dropout=dropout,
        encoder_type=encoder_type,
        pool_type=pool_type,
        n_heads=n_heads,
        use_assoc_aux=use_assoc_aux,
    )
