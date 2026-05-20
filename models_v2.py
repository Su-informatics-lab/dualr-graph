#!/usr/bin/env python3
"""
models_v2.py — Feature-graph autoencoder for ICI→AKI prediction.

Key design (informed by ChatGPT audit + empirical sweeps v4-v6):
  ① AKI node as query/target token — classify from z_aki, not mean(all)
  ② Decoupled topology (LLM edges) from association target (Pearson)
  ③ Static association loss on feature embeddings (not per-patient z)
  ④ WeightedDirGCNConv preserving LLM edge weights
  ⑤ Optional masked feature reconstruction (GraphMAE-style pretext)
  ⑥ Directed / symmetric / cross-correlation decoders
  ⑦ Optional VGAE, Laplacian PE

Readout modes (--readout):
  global_pool   : mean(all nodes)           — v4/v5 baseline
  non_aki_pool  : mean(non-AKI nodes)       — excludes masked node
  aki_only      : z_aki                     — outcome-node query
  aki_concat    : concat(z_aki, mean(non-AKI)) — recommended default
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, TransformerConv


@dataclass
class ModelOutput:
    z: torch.Tensor  # [B*F, emb]
    association_hat: torch.Tensor  # [B, F, F]
    aki_logits: torch.Tensor  # [B, n_classes]
    kl_loss: torch.Tensor | None
    recon_loss: torch.Tensor | None  # masked feature reconstruction


# ── Laplacian PE ─────────────────────────────────────────────


def compute_laplacian_pe(edge_index, n_nodes, pe_dim):
    adj = torch.zeros(n_nodes, n_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = (adj + adj.T).clamp(max=1.0)
    deg = adj.sum(1)
    deg_inv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    L_norm = torch.eye(n_nodes) - torch.diag(deg_inv) @ adj @ torch.diag(deg_inv)
    _, eigvecs = torch.linalg.eigh(L_norm)
    pe = eigvecs[:, 1 : pe_dim + 1]
    if pe.shape[1] < pe_dim:
        pe = F.pad(pe, (0, pe_dim - pe.shape[1]))
    return pe.float()


# ── WeightedDirGCNConv ───────────────────────────────────────


class WeightedDirGCNConv(nn.Module):
    """
    Direction-aware GCN that preserves edge weights.
    Runs separate convolutions on incoming and outgoing edges,
    combines with a learnable alpha.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_in = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
        self.conv_out = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
        self.root = nn.Linear(in_dim, out_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, edge_weight=None):
        a = torch.sigmoid(self.alpha)
        x_in = self.conv_in(x, edge_index, edge_weight=edge_weight)
        rev_ei = edge_index.flip(0)
        x_out = self.conv_out(x, rev_ei, edge_weight=edge_weight)
        return a * x_in + (1 - a) * x_out + self.root(x)


# ── Encoder blocks ───────────────────────────────────────────


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
        use_dir_conv=False,
    ):
        super().__init__()
        self.conv_type = conv_type.lower()
        self.use_dir_conv = use_dir_conv

        if use_dir_conv:
            # WeightedDirGCNConv — always GCN-based, preserves edge weights
            self.conv = WeightedDirGCNConv(in_dim, out_dim)
            actual_out = out_dim
        elif self.conv_type == "gcn":
            self.conv = GCNConv(in_dim, out_dim, normalize=True)
            actual_out = out_dim
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
        if self.use_dir_conv:
            ew = edge_attr.squeeze(-1) if edge_attr is not None else None
            x = self.conv(x, edge_index, edge_weight=ew)
        elif self.conv_type == "gcn":
            ew = edge_attr.squeeze(-1) if edge_attr is not None else None
            x = self.conv(x, edge_index, edge_weight=ew)
        else:
            x = self.conv(x, edge_index, edge_attr=edge_attr)
        return F.dropout(F.elu(self.norm(x)), p=self.dropout, training=self.training)


# ── Decoders ─────────────────────────────────────────────────


class SymmetricDecoder(nn.Module):
    def forward(self, z_by_graph):
        d = z_by_graph.size(-1)
        return torch.sigmoid(z_by_graph @ z_by_graph.transpose(1, 2) / math.sqrt(d))


class DirectedDecoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.proj_src = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_tgt = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, z_by_graph):
        z_src = self.proj_src(z_by_graph)
        z_tgt = self.proj_tgt(z_by_graph)
        return torch.sigmoid(
            z_tgt @ z_src.transpose(1, 2) / math.sqrt(z_by_graph.size(-1))
        )


class CrossCorrelationDecoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, z_by_graph):
        B, F, d = z_by_graph.shape
        z_i = z_by_graph.unsqueeze(2).expand(B, F, F, d)
        z_j = z_by_graph.unsqueeze(1).expand(B, F, F, d)
        return torch.sigmoid(self.proj(z_i * z_j).squeeze(-1))


# ── Masked Feature Reconstruction Head ───────────────────────


class MaskedFeatureReconHead(nn.Module):
    """Reconstruct masked node values from their embeddings."""

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_masked_nodes):
        return self.head(z_masked_nodes).squeeze(-1)


# ── Main Model ───────────────────────────────────────────────


class FeatureGraphAutoencoderV2(nn.Module):

    READOUT_MODES = ("global_pool", "non_aki_pool", "aki_only", "aki_concat")

    def __init__(
        self,
        n_features: int,
        node_input_dim: int = 2,
        hidden_dim: int = 64,
        embedding_dim: int = 16,
        n_layers: int = 2,
        encoder_type: str = "gcn",
        n_heads: int = 4,
        dropout: float = 0.1,
        n_classes: int = 2,
        edge_dim: int | None = None,
        aki_idx: int | None = None,
        decoder_type: str = "symmetric",
        use_vgae: bool = False,
        use_laplacian_pe: bool = False,
        pe_dim: int = 16,
        use_dir_conv: bool = False,
        readout: str = "aki_concat",
        use_masked_recon: bool = False,
    ):
        super().__init__()
        assert readout in self.READOUT_MODES, f"readout={readout!r}"
        self.n_features = n_features
        self.aki_idx = aki_idx
        self.readout = readout
        self.use_vgae = use_vgae
        self.use_laplacian_pe = use_laplacian_pe
        self.use_masked_recon = use_masked_recon
        self.pe_dim = pe_dim

        # Feature identity embedding
        identity_dim = hidden_dim
        self.feature_embedding = nn.Embedding(n_features, identity_dim)

        # Laplacian PE
        pe_proj_dim = 0
        if use_laplacian_pe:
            pe_proj_dim = hidden_dim // 4
            self.pe_proj = nn.Linear(pe_dim, pe_proj_dim)
            self.register_buffer("_lap_pe", torch.zeros(n_features, pe_dim))

        self.input_proj = nn.Linear(
            node_input_dim + identity_dim + pe_proj_dim, hidden_dim
        )

        # Encoder
        blocks = []
        current = hidden_dim
        for i in range(n_layers):
            is_last = i == n_layers - 1
            od = embedding_dim if is_last else hidden_dim
            oh = 1 if is_last else n_heads
            cc = not is_last
            blocks.append(
                GraphBlock(
                    encoder_type,
                    current,
                    od,
                    oh,
                    cc,
                    dropout,
                    edge_dim,
                    use_dir_conv=use_dir_conv,
                )
            )
            current = blocks[-1].actual_out
        self.encoder = nn.ModuleList(blocks)

        # VGAE
        if use_vgae:
            self.mu_head = nn.Linear(embedding_dim, embedding_dim)
            self.logvar_head = nn.Linear(embedding_dim, embedding_dim)

        # Decoder (operates on per-patient z or static embeddings)
        if decoder_type == "symmetric":
            self.decoder = SymmetricDecoder()
        elif decoder_type == "directed":
            self.decoder = DirectedDecoder(embedding_dim)
        elif decoder_type == "crosscorr":
            self.decoder = CrossCorrelationDecoder(embedding_dim)
        else:
            raise ValueError(f"decoder_type={decoder_type!r}")

        # AKI head — input dim depends on readout mode
        if readout in ("global_pool", "non_aki_pool", "aki_only"):
            aki_in = embedding_dim
        elif readout == "aki_concat":
            aki_in = embedding_dim * 2
        self.aki_head = nn.Sequential(
            nn.Linear(aki_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        # Masked feature reconstruction
        if use_masked_recon:
            self.recon_head = MaskedFeatureReconHead(embedding_dim, hidden_dim)

    def register_laplacian_pe(self, edge_index):
        pe = compute_laplacian_pe(edge_index, self.n_features, self.pe_dim)
        self._lap_pe = pe.to(self._lap_pe.device)

    def encode(self, x, edge_index, feature_id, edge_attr=None):
        parts = [x, self.feature_embedding(feature_id)]
        if self.use_laplacian_pe:
            parts.append(self.pe_proj(self._lap_pe[feature_id]))
        h = F.relu(self.input_proj(torch.cat(parts, dim=-1)))
        for block in self.encoder:
            h = block(h, edge_index, edge_attr)
        return h

    def reparameterise(self, z_flat):
        mu = self.mu_head(z_flat)
        logvar = self.logvar_head(z_flat)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std if self.training else mu
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl

    def _readout(self, z_by_graph):
        """Extract classification features from per-node embeddings."""
        if self.readout == "global_pool":
            return z_by_graph.mean(dim=1)
        elif self.readout == "non_aki_pool":
            mask = torch.ones(
                self.n_features, dtype=torch.bool, device=z_by_graph.device
            )
            mask[self.aki_idx] = False
            return z_by_graph[:, mask, :].mean(dim=1)
        elif self.readout == "aki_only":
            return z_by_graph[:, self.aki_idx, :]
        elif self.readout == "aki_concat":
            z_aki = z_by_graph[:, self.aki_idx, :]
            mask = torch.ones(
                self.n_features, dtype=torch.bool, device=z_by_graph.device
            )
            mask[self.aki_idx] = False
            z_ctx = z_by_graph[:, mask, :].mean(dim=1)
            return torch.cat([z_aki, z_ctx], dim=-1)

    def static_association(self):
        """Association from feature identity embeddings (patient-invariant)."""
        E = self.feature_embedding.weight  # [F, identity_dim]
        # Project to embedding_dim via the first encoder block's input path
        # For simplicity, use the decoder on a pseudo-z from identity embeddings
        # The decoder expects [1, F, emb_dim], so we need a projection
        # Use a simple inner product on the raw embeddings
        d = E.size(-1)
        return torch.sigmoid(E @ E.T / math.sqrt(d)).unsqueeze(0)

    def forward(self, batch, pretext_mask=None) -> ModelOutput:
        z_flat = self.encode(
            batch.x, batch.edge_index, batch.feature_id, batch.edge_attr
        )

        kl_loss = None
        if self.use_vgae:
            z_flat, kl_loss = self.reparameterise(z_flat)

        z_by_graph = z_flat.view(batch.num_graphs, self.n_features, -1)

        # Association decoder (per-patient or static — chosen externally via loss)
        association_hat = self.decoder(z_by_graph)

        # AKI prediction from readout
        cls_features = self._readout(z_by_graph)
        logits = self.aki_head(cls_features)

        # Masked feature reconstruction
        recon_loss = None
        if self.use_masked_recon and pretext_mask is not None and self.training:
            # pretext_mask: [B*F] bool — True for masked observed non-AKI nodes
            if pretext_mask.any():
                z_masked = z_flat[pretext_mask]
                recon_loss = z_masked  # raw embeddings; loss computed externally

        return ModelOutput(
            z=z_flat,
            association_hat=association_hat,
            aki_logits=logits,
            kl_loss=kl_loss,
            recon_loss=recon_loss,
        )


# ── Loss helpers ─────────────────────────────────────────────


def association_mse(a_hat, a_target, ignore_diag=True, directed=False):
    target = a_target.to(a_hat.device).unsqueeze(0).expand_as(a_hat)
    if not directed:
        a_hat = (a_hat + a_hat.transpose(1, 2)) / 2
        target = (target + target.transpose(1, 2)) / 2
    n = target.size(-1)
    if ignore_diag:
        mask = ~torch.eye(n, dtype=torch.bool, device=target.device)
        return F.mse_loss(a_hat[:, mask], target[:, mask])
    return F.mse_loss(a_hat, target)


def static_association_mse(model, a_target, ignore_diag=True):
    """Association loss on feature identity embeddings (patient-invariant)."""
    a_hat = model.static_association()  # [1, F, F]
    target = a_target.to(a_hat.device).unsqueeze(0)
    n = target.size(-1)
    # Always symmetric for static
    a_hat = (a_hat + a_hat.transpose(1, 2)) / 2
    target = (target + target.transpose(1, 2)) / 2
    if ignore_diag:
        mask = ~torch.eye(n, dtype=torch.bool, device=target.device)
        return F.mse_loss(a_hat[:, mask], target[:, mask])
    return F.mse_loss(a_hat, target)


# ── Builder ──────────────────────────────────────────────────


def build_model(
    n_features,
    aki_idx,
    hidden_dim=64,
    embedding_dim=16,
    n_layers=2,
    encoder_type="gcn",
    n_heads=4,
    dropout=0.1,
    n_classes=2,
    edge_dim=None,
    decoder_type="symmetric",
    use_vgae=False,
    use_laplacian_pe=False,
    pe_dim=16,
    use_dir_conv=False,
    readout="aki_concat",
    use_masked_recon=False,
    **_,
):
    return FeatureGraphAutoencoderV2(
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
        aki_idx=aki_idx,
        decoder_type=decoder_type,
        use_vgae=use_vgae,
        use_laplacian_pe=use_laplacian_pe,
        pe_dim=pe_dim,
        use_dir_conv=use_dir_conv,
        readout=readout,
        use_masked_recon=use_masked_recon,
    )
