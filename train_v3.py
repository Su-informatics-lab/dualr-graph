#!/usr/bin/env python3
"""
models_v3.py — Feature-graph autoencoder for ICI→AKI prediction.

v10.4 anti-collapse fixes:
  ① Residual message-passing blocks: h_out = LayerNorm(skip(h_in) + dropout(ELU(conv(h_in))))
  ② Jumping Knowledge: concatenate early-layer embeddings before final projection.
  ③ Target-sum normalization for directed LLM edge weights to prevent message explosion.
  ④ Optional DropEdge for structural regularization.
  ⑤ Separate AKI attention head, so AKI prediction is not forced to come only from x_hat_aki.
  ⑥ Contextual value decoder retained for reconstruction auxiliary.
  ⑦ Optional association decoder retained for the proven v5-style regularizer.

Default recommended mode is hybrid:
  - directed LLM topology for message passing
  - association auxiliary as the main structural regularizer
  - small value-reconstruction auxiliary
  - AKI prediction from attention over non-AKI nodes + z_AKI
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
    z: torch.Tensor  # [B*F, emb]
    x_hat: torch.Tensor  # [B, F]
    aki_prob: torch.Tensor  # [B]
    aki_logit: torch.Tensor  # [B]
    association_hat: torch.Tensor | None = None
    attention: torch.Tensor | None = None  # [B, F-1], only for attention readout


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def drop_edges(
    edge_index: torch.Tensor, edge_weight: torch.Tensor | None, p: float, training: bool
):
    if (not training) or p <= 0:
        return edge_index, edge_weight
    n_edges = edge_index.size(1)
    if n_edges == 0:
        return edge_index, edge_weight
    keep = torch.rand(n_edges, device=edge_index.device) > p
    if keep.sum() == 0:
        keep[torch.randint(n_edges, (1,), device=edge_index.device)] = True
    edge_index = edge_index[:, keep]
    if edge_weight is not None:
        edge_weight = edge_weight[keep]
    return edge_index, edge_weight


def target_sum_normalize(
    edge_index: torch.Tensor, edge_weight: torch.Tensor | None, n_nodes: int
):
    """Normalize incoming edge weights so messages into each target sum to 1."""
    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device, dtype=torch.float32
        )
    if edge_weight.dim() > 1:
        edge_weight = edge_weight.view(-1)

    target = edge_index[1]
    denom = torch.zeros(n_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    denom.index_add_(0, target, edge_weight)
    return edge_weight / denom[target].clamp(min=1e-12)


class PairNorm(nn.Module):
    """Light PairNorm over feature nodes within each patient graph."""

    def __init__(self, mode: str = "none", scale: float = 1.0, eps: float = 1e-6):
        super().__init__()
        if mode not in ("none", "center", "scale"):
            raise ValueError("pairnorm must be one of: none, center, scale")
        self.mode = mode
        self.scale = scale
        self.eps = eps

    def forward(self, x: torch.Tensor, batch_size: int, n_features: int):
        if self.mode == "none":
            return x
        z = x.view(batch_size, n_features, -1)
        z = z - z.mean(dim=1, keepdim=True)
        if self.mode == "scale":
            row_norm = (
                z.pow(2)
                .sum(dim=-1)
                .mean(dim=1, keepdim=True)
                .sqrt()
                .clamp(min=self.eps)
            )
            z = self.scale * z / row_norm.unsqueeze(-1)
        return z.reshape_as(x)


# ─────────────────────────────────────────────────────────────
# Directed weighted message passing
# ─────────────────────────────────────────────────────────────


class DirectedWeightedConv(MessagePassing):
    """v_i^(l+1) = W_self·v_i^(l) + Σ W_neigh·v_j^(l)·P(v_i|v_j)."""

    def __init__(self, in_dim: int, out_dim: int, edge_norm: str = "target_sum"):
        super().__init__(aggr="add", flow="source_to_target")
        if edge_norm not in ("none", "target_sum"):
            raise ValueError("edge_norm must be 'none' or 'target_sum'")
        self.edge_norm = edge_norm
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.view(-1)
        if self.edge_norm == "target_sum":
            edge_weight = target_sum_normalize(edge_index, edge_weight, x.size(0))
        return self.w_self(x) + self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight=None):
        msg = self.w_neigh(x_j)
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg


# ─────────────────────────────────────────────────────────────
# Encoder blocks
# ─────────────────────────────────────────────────────────────


class DirectedGCNBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout=0.1, residual=True, edge_norm="target_sum"
    ):
        super().__init__()
        self.conv = DirectedWeightedConv(in_dim, out_dim, edge_norm=edge_norm)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        h_msg = self.conv(x, edge_index, edge_weight)
        h_msg = F.dropout(F.elu(h_msg), p=self.dropout, training=self.training)
        if self.residual:
            return self.norm(self.skip(x) + h_msg)
        return F.dropout(
            F.elu(self.norm(h_msg)), p=self.dropout, training=self.training
        )


class GATv2Block(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        heads=4,
        dropout=0.1,
        edge_dim=None,
        residual=True,
    ):
        super().__init__()
        head_out = max(1, out_dim // heads)
        self.conv = GATv2Conv(
            in_dim,
            head_out,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
            residual=True,
        )
        actual_out = head_out * heads
        self.proj = (
            nn.Linear(actual_out, out_dim) if actual_out != out_dim else nn.Identity()
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None
        h_msg = self.proj(self.conv(x, edge_index, edge_attr=edge_attr))
        h_msg = F.dropout(F.elu(h_msg), p=self.dropout, training=self.training)
        if self.residual:
            return self.norm(self.skip(x) + h_msg)
        return F.dropout(
            F.elu(self.norm(h_msg)), p=self.dropout, training=self.training
        )


# ─────────────────────────────────────────────────────────────
# Decoders and heads
# ─────────────────────────────────────────────────────────────


class ContextualValueDecoder(nn.Module):
    """decoder(z_j, global_context) → x̂_j."""

    POOL_TYPES = ("mean", "max", "mean_max")

    def __init__(self, embedding_dim: int, hidden_dim: int, pool_type: str = "mean"):
        super().__init__()
        if pool_type not in self.POOL_TYPES:
            raise ValueError(f"pool_type must be one of {self.POOL_TYPES}")
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
        z_bg = z.view(batch_size, n_features, -1)
        if self.pool_type == "mean":
            ctx = z_bg.mean(dim=1)
        elif self.pool_type == "max":
            ctx = z_bg.max(dim=1).values
        else:
            ctx = torch.cat([z_bg.mean(dim=1), z_bg.max(dim=1).values], dim=-1)
        ctx_exp = ctx.unsqueeze(1).expand(-1, n_features, -1)
        combined = torch.cat([z_bg, ctx_exp], dim=-1)
        return self.mlp(combined.reshape(-1, combined.size(-1))).view(
            batch_size, n_features
        )


class SymmetricAssociationDecoder(nn.Module):
    def forward(self, z_by_graph):
        d = z_by_graph.size(-1)
        return torch.sigmoid(z_by_graph @ z_by_graph.transpose(1, 2) / math.sqrt(d))


class AKIAttentionHead(nn.Module):
    """Predict AKI from z_AKI plus attention over non-AKI feature nodes."""

    READOUTS = ("attention", "concat")

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_features: int,
        aki_idx: int,
        mode: str = "attention",
    ):
        super().__init__()
        if mode not in self.READOUTS:
            raise ValueError(f"mode must be one of {self.READOUTS}")
        self.mode = mode
        self.n_features = n_features
        self.aki_idx = aki_idx
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if mode == "attention":
            in_dim = 3 * embedding_dim  # z_aki, attended context, mean context
        else:
            in_dim = 3 * embedding_dim  # z_aki, mean context, max context
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_by_graph: torch.Tensor):
        mask = torch.ones(self.n_features, dtype=torch.bool, device=z_by_graph.device)
        mask[self.aki_idx] = False
        z_aki = z_by_graph[:, self.aki_idx, :]
        z_other = z_by_graph[:, mask, :]

        mean_ctx = z_other.mean(dim=1)
        if self.mode == "concat":
            max_ctx = z_other.max(dim=1).values
            logit = self.mlp(torch.cat([z_aki, mean_ctx, max_ctx], dim=-1)).squeeze(-1)
            return logit, None

        q = self.query(z_aki).unsqueeze(1)
        k = self.key(z_other)
        v = self.value(z_other)
        scores = (q * k).sum(dim=-1) / math.sqrt(z_by_graph.size(-1))
        attn = torch.softmax(scores, dim=-1)
        attn_ctx = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        logit = self.mlp(torch.cat([z_aki, attn_ctx, mean_ctx], dim=-1)).squeeze(-1)
        return logit, attn


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────


class FeatureGraphAutoencoderV3(nn.Module):
    AKI_READOUTS = ("reconstruct", "attention", "concat")
    JK_MODES = ("last", "concat", "max")

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
        residual: bool = True,
        jk_mode: str = "concat",
        edge_norm: str = "target_sum",
        edge_dropout: float = 0.0,
        pairnorm: str = "none",
        aki_readout: str = "attention",
    ):
        super().__init__()
        if jk_mode not in self.JK_MODES:
            raise ValueError(f"jk_mode must be one of {self.JK_MODES}")
        if aki_readout not in self.AKI_READOUTS:
            raise ValueError(f"aki_readout must be one of {self.AKI_READOUTS}")

        self.n_features = n_features
        self.aki_idx = aki_idx
        self.embedding_dim = embedding_dim
        self.use_assoc_aux = use_assoc_aux
        self.jk_mode = jk_mode
        self.edge_dropout = edge_dropout
        self.aki_readout = aki_readout
        self.encoder_type = encoder_type

        self.feature_embedding = nn.Embedding(n_features, hidden_dim)
        self.input_proj = nn.Linear(2 + hidden_dim, hidden_dim)
        self.pairnorm = PairNorm(pairnorm)

        blocks = []
        for _ in range(n_layers):
            if encoder_type == "directed":
                blocks.append(
                    DirectedGCNBlock(
                        hidden_dim,
                        hidden_dim,
                        dropout=dropout,
                        residual=residual,
                        edge_norm=edge_norm,
                    )
                )
            elif encoder_type == "gatv2":
                blocks.append(
                    GATv2Block(
                        hidden_dim,
                        hidden_dim,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=1,
                        residual=residual,
                    )
                )
            else:
                raise ValueError(f"Unknown encoder_type={encoder_type}")
        self.encoder = nn.ModuleList(blocks)

        if jk_mode == "concat":
            self.jk_proj = nn.Linear(hidden_dim * n_layers, embedding_dim)
        else:
            self.jk_proj = nn.Linear(hidden_dim, embedding_dim)

        self.value_decoder = ContextualValueDecoder(
            embedding_dim, hidden_dim, pool_type
        )
        if use_assoc_aux:
            self.assoc_decoder = SymmetricAssociationDecoder()

        if aki_readout in ("attention", "concat"):
            self.aki_head = AKIAttentionHead(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                n_features=n_features,
                aki_idx=aki_idx,
                mode=("attention" if aki_readout == "attention" else "concat"),
            )

    def encode(
        self, x, edge_index, feature_id, edge_weight=None, batch_size: int | None = None
    ):
        identity = self.feature_embedding(feature_id)
        h = F.relu(self.input_proj(torch.cat([x, identity], dim=-1)))

        edge_index_use, edge_weight_use = drop_edges(
            edge_index, edge_weight, self.edge_dropout, self.training
        )

        layer_outputs = []
        for block in self.encoder:
            h = block(h, edge_index_use, edge_weight_use)
            if batch_size is not None:
                h = self.pairnorm(h, batch_size, self.n_features)
            layer_outputs.append(h)

        if self.jk_mode == "concat":
            h_jk = torch.cat(layer_outputs, dim=-1)
        elif self.jk_mode == "max":
            h_jk = torch.stack(layer_outputs, dim=0).max(dim=0).values
        else:
            h_jk = layer_outputs[-1]
        return self.jk_proj(h_jk)

    def decode(self, z, batch_size):
        return self.value_decoder(z, batch_size, self.n_features)

    def forward(self, batch, n_iter: int = 1) -> ModelOutput:
        edge_weight = getattr(batch, "edge_weight", None)
        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.view(-1)
        x_current = batch.x

        for iteration in range(n_iter):
            z = self.encode(
                x_current,
                batch.edge_index,
                batch.feature_id,
                edge_weight,
                batch_size=batch.num_graphs,
            )
            x_hat = self.decode(z, batch.num_graphs)
            if iteration < n_iter - 1:
                x_filled = x_current.clone()
                aki_pos = (batch.feature_id == self.aki_idx).nonzero(as_tuple=True)[0]
                x_filled[aki_pos, 0] = torch.sigmoid(x_hat[:, self.aki_idx]).detach()
                x_filled[aki_pos, 1] = 1.0
                x_current = x_filled

        z_by_graph = z.view(batch.num_graphs, self.n_features, -1)
        attention = None
        if self.aki_readout == "reconstruct":
            aki_logit = x_hat[:, self.aki_idx]
        else:
            aki_logit, attention = self.aki_head(z_by_graph)
        aki_prob = torch.sigmoid(aki_logit)

        assoc = None
        if self.use_assoc_aux:
            assoc = self.assoc_decoder(z_by_graph)

        return ModelOutput(
            z=z,
            x_hat=x_hat,
            aki_prob=aki_prob,
            aki_logit=aki_logit,
            association_hat=assoc,
            attention=attention,
        )


# ─────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────


def reconstruction_mse(x_hat, x_true, obs_mask):
    """MSE over observed entries selected by obs_mask."""
    diff_sq = (x_hat - x_true).pow(2)
    masked = diff_sq * obs_mask
    return masked.sum() / obs_mask.sum().clamp(min=1)


def aki_ce_loss(aki_logit, aki_true):
    return F.binary_cross_entropy_with_logits(
        aki_logit, aki_true.float(), reduction="mean"
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
    out: ModelOutput,
    x_true,
    obs_mask,
    aki_true,
    assoc_target=None,
    lambda_rec=0.1,
    lambda_aki=1.0,
    lambda_assoc=1.0,
):
    l_rec = reconstruction_mse(out.x_hat, x_true, obs_mask)
    l_aki = aki_ce_loss(out.aki_logit, aki_true)
    l_assoc = torch.tensor(0.0, device=out.x_hat.device)
    if (
        assoc_target is not None
        and out.association_hat is not None
        and lambda_assoc > 0
    ):
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
    residual: bool = True,
    jk_mode: str = "concat",
    edge_norm: str = "target_sum",
    edge_dropout: float = 0.0,
    pairnorm: str = "none",
    aki_readout: str = "attention",
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
        residual=residual,
        jk_mode=jk_mode,
        edge_norm=edge_norm,
        edge_dropout=edge_dropout,
        pairnorm=pairnorm,
        aki_readout=aki_readout,
    )
