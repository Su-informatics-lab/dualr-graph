#!/usr/bin/env python3
"""
models.py — Feature-graph autoencoder for ICI-AKI prediction.

v3.1 fixes (GPT deep search 2026-05-19):
  ① Feature-type embedding (binary vs continuous) in node input
  ② binary_mask buffer for type-aware reconstruction loss
  ③ AKI query node support (no label leakage by design)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, TransformerConv


@dataclass
class ModelOutput:
    x_hat: torch.Tensor  # [B, F]
    aki_logits: torch.Tensor  # [B]
    z: torch.Tensor  # [B, F, latent]


def _repeat_graph(edge_index, num_graphs, num_nodes):
    if num_graphs == 1:
        return edge_index
    edge_index = edge_index.long()
    ne = edge_index.size(1)
    off = (
        torch.arange(num_graphs, device=edge_index.device)
        .repeat_interleave(ne)
        .mul(num_nodes)
    )
    return edge_index.repeat(1, num_graphs) + off.unsqueeze(0)


def _repeat_edge_attr(edge_attr, num_graphs):
    if edge_attr is None:
        return None
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)
    return edge_attr.repeat(num_graphs, 1)


def drop_edge(edge_index, edge_attr, p, training):
    if not training or p <= 0:
        return edge_index, edge_attr
    mask = torch.rand(edge_index.size(1), device=edge_index.device) >= p
    return edge_index[:, mask], (edge_attr[mask] if edge_attr is not None else None)


class GraphBlock(nn.Module):
    def __init__(
        self,
        conv_type,
        in_dim,
        out_dim,
        heads,
        concat,
        dropout,
        edge_dim,
        add_self_loops=True,
    ):
        super().__init__()
        self.conv_type = conv_type.lower()
        if self.conv_type == "gcn":
            self.conv = GCNConv(
                in_dim, out_dim, add_self_loops=add_self_loops, normalize=True
            )
            actual_out = out_dim
        elif self.conv_type == "gat":
            self.conv = GATConv(
                in_dim,
                out_dim,
                heads=heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=add_self_loops,
                fill_value=1.0,
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
                add_self_loops=add_self_loops,
                fill_value=1.0,
                residual=True,
            )
            actual_out = out_dim * heads if concat else out_dim
        elif self.conv_type == "transformer":
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
        x = self.norm(x)
        x = F.elu(x)
        return F.dropout(x, p=self.dropout, training=self.training)


class UncertaintyWeighting(nn.Module):
    """Kendall et al. CVPR 2018: learnable task weights via homoscedastic uncertainty."""

    def __init__(self):
        super().__init__()
        self.log_var_rec = nn.Parameter(torch.zeros(1))
        self.log_var_aki = nn.Parameter(torch.zeros(1))

    def forward(self, loss_rec, loss_aki):
        w_rec = torch.exp(-self.log_var_rec)
        w_aki = torch.exp(-self.log_var_aki)
        return (
            0.5 * w_rec * loss_rec
            + 0.5 * self.log_var_rec
            + 0.5 * w_aki * loss_aki
            + 0.5 * self.log_var_aki
        ).squeeze()


class FeatureGraphAE(nn.Module):
    """
    Patient-as-graph-signal feature-graph autoencoder.

    AKI modes:
      dual   : MLP(z_K ‖ z̄) — encoder head (GRAPE-style, default)
      pooled : MLP(attn_pool(Z)) — graph-level readout (INCE/T2G style)
      recon_split : Linear_aki(d_K) — decoder head, own projection
      recon  : shared recon_head (baseline, known to fail)
    """

    def __init__(
        self,
        num_features,
        aki_idx,
        node_emb_dim=8,
        hidden=64,
        latent=32,
        heads=4,
        layers=2,
        dropout=0.15,
        edge_dim=1,
        conv_type="gatv2",
        aki_mode="dual",
        add_self_loops=True,
        drop_edge_rate=0.0,
    ):
        super().__init__()
        if aki_mode not in ("recon", "recon_split", "dual", "pooled"):
            raise ValueError(f"aki_mode={aki_mode!r}")

        self.num_features = int(num_features)
        self.aki_idx = int(aki_idx)
        self.aki_mode = aki_mode
        self.drop_edge_rate = drop_edge_rate

        # Fix #4: feature-id + feature-type embeddings
        self.node_emb = nn.Embedding(num_features, node_emb_dim)
        self.feat_type_emb = nn.Embedding(2, 4)  # 0=binary, 1=continuous
        # binary_mask stored via register_binary_mask() after construction
        self.register_buffer(
            "_binary_mask", torch.zeros(num_features, dtype=torch.long)
        )

        # value(1) + missing(1) + aki_flag(1) + id_emb(node_emb_dim) + type_emb(4)
        in_dim = 1 + 1 + 1 + node_emb_dim + 4

        # ── Encoder ──────────────────────────────────────────────
        enc_blocks = []
        current = in_dim
        for i in range(layers):
            is_last = i == layers - 1
            od = latent if is_last else hidden
            oh = 1 if is_last else heads
            cc = not is_last
            enc_blocks.append(
                GraphBlock(
                    conv_type, current, od, oh, cc, dropout, edge_dim, add_self_loops
                )
            )
            current = enc_blocks[-1].actual_out
        self.encoder = nn.ModuleList(enc_blocks)

        # ── GNN Decoder ──────────────────────────────────────────
        self.dec1 = GraphBlock(
            conv_type, latent, hidden, heads, True, dropout, edge_dim, add_self_loops
        )
        self.dec2 = GraphBlock(
            conv_type,
            self.dec1.actual_out,
            hidden,
            1,
            False,
            dropout,
            edge_dim,
            add_self_loops,
        )
        self.recon_head = nn.Linear(hidden, 1)

        # ── AKI heads ────────────────────────────────────────────
        self.aki_recon_head = nn.Linear(hidden, 1)
        self.aki_head = nn.Sequential(
            nn.Linear(latent * 2, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.attn_pool = nn.Linear(latent, 1)
        self.aki_pool_head = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def register_binary_mask(self, binary_mask):
        """Call after construction: binary_mask[j]=True if feature j is binary."""
        # Store as long tensor: 0=binary, 1=continuous (for feat_type_emb)
        self._binary_mask = (~binary_mask.bool()).long().to(self._binary_mask.device)

    def _make_node_input(self, x, missing_mask):
        B, F = x.shape
        device = x.device
        node_ids = torch.arange(F, device=device).repeat(B)
        values = x.reshape(B * F, 1)
        miss = (
            missing_mask.reshape(B * F, 1).float()
            if missing_mask is not None
            else torch.zeros_like(values)
        )
        is_aki = (node_ids == self.aki_idx).float().unsqueeze(-1)
        id_emb = self.node_emb(node_ids)
        # Fix #4: feature-type embedding
        type_ids = self._binary_mask.to(device).repeat(B)  # [B*F]
        type_emb = self.feat_type_emb(type_ids)  # [B*F, 4]
        return torch.cat([values, miss, is_aki, id_emb, type_emb], dim=-1)

    def encode(self, x, edge_index, edge_attr=None, missing_mask=None):
        B, F = x.shape
        ei = _repeat_graph(edge_index.to(x.device), B, F)
        ea = _repeat_edge_attr(
            edge_attr.to(x.device) if edge_attr is not None else None, B
        )
        ei, ea = drop_edge(ei, ea, self.drop_edge_rate, self.training)
        h = self._make_node_input(x, missing_mask)
        for block in self.encoder:
            h = block(h, ei, ea)
        return h.view(B, F, -1), ei, ea

    def decode(self, h_flat, ei, ea, B, F):
        d = self.dec1(h_flat, ei, ea)
        d = self.dec2(d, ei, ea)
        d_nodes = d.view(B, F, -1)
        x_hat = self.recon_head(d).view(B, F)
        return x_hat, d_nodes

    def forward(self, x, edge_index, edge_attr=None, missing_mask=None):
        B, F = x.shape
        z, ei, ea = self.encode(x, edge_index, edge_attr, missing_mask)
        x_hat, d_nodes = self.decode(z.reshape(B * F, -1), ei, ea, B, F)

        if self.aki_mode == "recon":
            aki = x_hat[:, self.aki_idx]
        elif self.aki_mode == "recon_split":
            aki = self.aki_recon_head(d_nodes[:, self.aki_idx, :]).squeeze(-1)
            x_hat = x_hat.clone()
            x_hat[:, self.aki_idx] = aki
        elif self.aki_mode == "dual":
            aki = self.aki_head(
                torch.cat([z[:, self.aki_idx, :], z.mean(1)], -1)
            ).squeeze(-1)
        elif self.aki_mode == "pooled":
            a = torch.softmax(self.attn_pool(z).squeeze(-1), dim=-1)
            aki = self.aki_pool_head((a.unsqueeze(-1) * z).sum(1)).squeeze(-1)

        return ModelOutput(x_hat=x_hat, aki_logits=aki, z=z)

    @torch.no_grad()
    def predict_proba(self, x, edge_index, edge_attr=None, missing_mask=None):
        self.eval()
        return torch.sigmoid(
            self.forward(x, edge_index, edge_attr, missing_mask).aki_logits
        )

    @torch.no_grad()
    def extract_embeddings(self, x, edge_index, edge_attr=None, missing_mask=None):
        self.eval()
        z, _, _ = self.encode(x, edge_index, edge_attr, missing_mask)
        return torch.cat([z[:, self.aki_idx, :], z.mean(1)], -1).cpu().numpy()


def build_model(
    num_features=38,
    aki_idx=37,
    node_emb_dim=8,
    hidden=64,
    latent=32,
    heads=4,
    layers=2,
    dropout=0.15,
    edge_dim=1,
    conv_type="gatv2",
    aki_mode="dual",
    add_self_loops=True,
    drop_edge_rate=0.0,
    **_,
):
    return FeatureGraphAE(
        num_features=num_features,
        aki_idx=aki_idx,
        node_emb_dim=node_emb_dim,
        hidden=hidden,
        latent=latent,
        heads=heads,
        layers=layers,
        dropout=dropout,
        edge_dim=edge_dim,
        conv_type=conv_type,
        aki_mode=aki_mode,
        add_self_loops=add_self_loops,
        drop_edge_rate=drop_edge_rate,
    )
