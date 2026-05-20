#!/usr/bin/env python3
"""
models_v2.py — Feature-graph autoencoder with directed association decoder.

Key upgrades over models.py (v4):
  ① Directed decoder: separate src/tgt projections so P(i|j) ≠ P(j|i)
  ② Optional VGAE: μ/σ heads + KL regularisation on latent
  ③ Laplacian Positional Encoding for structural awareness
  ④ Cross-correlation decoder option (S2GAE-inspired)

References:
  - Gravity-Inspired GAE (Salha et al., ECML-PKDD 2019)
  - S2GAE (Tan et al., WSDM 2023)
  - VGAE (Kipf & Welling, NeurIPS 2016 Workshop)
  - GraphGPS LapPE (Rampášek et al., NeurIPS 2022)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, TransformerConv, global_mean_pool

# ── Dataclass ────────────────────────────────────────────────


@dataclass
class ModelOutput:
    z: torch.Tensor  # [B, F, emb]
    association_hat: torch.Tensor  # [B, F, F]
    aki_logits: torch.Tensor  # [B, n_classes]
    kl_loss: torch.Tensor | None  # scalar, only if VGAE


# ── Laplacian Positional Encoding ────────────────────────────


def compute_laplacian_pe(
    edge_index: torch.Tensor, n_nodes: int, pe_dim: int
) -> torch.Tensor:
    """
    Precompute Laplacian eigenvector PE for a small graph.
    For 38 nodes, we can use all non-trivial eigenvectors.
    Returns: [n_nodes, pe_dim] float tensor.
    """
    # Build adjacency from edge_index (undirected for Laplacian)
    adj = torch.zeros(n_nodes, n_nodes)
    src, tgt = edge_index[0], edge_index[1]
    adj[src, tgt] = 1.0
    adj = (adj + adj.T).clamp(max=1.0)  # symmetrise for Laplacian
    deg = adj.sum(dim=1)
    D = torch.diag(deg)
    L = D - adj
    # Normalised Laplacian
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L_norm = torch.eye(n_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt

    eigvals, eigvecs = torch.linalg.eigh(L_norm)
    # Skip first eigenvector (constant), take next pe_dim
    pe = eigvecs[:, 1 : pe_dim + 1]  # [n_nodes, pe_dim]
    # Sign ambiguity: random sign flip during training handled externally
    if pe.shape[1] < pe_dim:
        pe = F.pad(pe, (0, pe_dim - pe.shape[1]))
    return pe.float()


# ── Encoder blocks ───────────────────────────────────────────


class GraphBlock(nn.Module):
    def __init__(self, conv_type, in_dim, out_dim, heads, concat, dropout, edge_dim):
        super().__init__()
        self.conv_type = conv_type.lower()
        if self.conv_type == "gcn":
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
        if self.conv_type == "gcn":
            ew = edge_attr.squeeze(-1) if edge_attr is not None else None
            x = self.conv(x, edge_index, edge_weight=ew)
        else:
            x = self.conv(x, edge_index, edge_attr=edge_attr)
        return F.dropout(F.elu(self.norm(x)), p=self.dropout, training=self.training)


# ── Decoders ─────────────────────────────────────────────────


class SymmetricDecoder(nn.Module):
    """Original inner-product: sigmoid(Z @ Z.T / sqrt(d))."""

    def forward(self, z_by_graph: torch.Tensor) -> torch.Tensor:
        d = z_by_graph.size(-1)
        scores = torch.matmul(z_by_graph, z_by_graph.transpose(1, 2))
        return torch.sigmoid(scores / math.sqrt(d))


class DirectedDecoder(nn.Module):
    """
    Asymmetric decoder: sigmoid(Z_tgt @ Z_src.T / sqrt(d)).

    Each node produces two representations:
      z_src: "I am a source (conditioning variable)"
      z_tgt: "I am a target (being predicted)"

    A_hat[i,j] = sigmoid(z_tgt_i · z_src_j / sqrt(d)) ≈ P(i|j)

    This respects the LLM prior's directionality.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.proj_src = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_tgt = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, z_by_graph: torch.Tensor) -> torch.Tensor:
        z_src = self.proj_src(z_by_graph)  # [B, F, d]
        z_tgt = self.proj_tgt(z_by_graph)  # [B, F, d]
        d = z_by_graph.size(-1)
        scores = torch.matmul(z_tgt, z_src.transpose(1, 2))  # [B, F, F]
        return torch.sigmoid(scores / math.sqrt(d))


class CrossCorrelationDecoder(nn.Module):
    """
    S2GAE-inspired: element-wise correlation of paired embeddings,
    reduced to a scalar per pair via a learned projection.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, z_by_graph: torch.Tensor) -> torch.Tensor:
        B, F, d = z_by_graph.shape
        # Pairwise element-wise product: z_i ⊙ z_j for all (i,j)
        z_i = z_by_graph.unsqueeze(2).expand(B, F, F, d)
        z_j = z_by_graph.unsqueeze(1).expand(B, F, F, d)
        cross = z_i * z_j  # [B, F, F, d]
        scores = self.proj(cross).squeeze(-1)  # [B, F, F]
        return torch.sigmoid(scores)


# ── Main model ───────────────────────────────────────────────


class FeatureGraphAutoencoderV2(nn.Module):
    """
    Feature-graph autoencoder with directed decoder + optional VGAE.

    Decoder modes:
      symmetric  : z @ z.T  (baseline, for Spearman/MI targets)
      directed   : z_tgt @ z_src.T  (for LLM directed targets)
      crosscorr  : proj(z_i ⊙ z_j)  (S2GAE-inspired)

    VGAE mode:
      Adds μ/σ projection heads after encoder → reparameterised z.
      KL divergence returned as auxiliary loss.
    """

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
        # New v2 options
        decoder_type: str = "directed",  # symmetric / directed / crosscorr
        use_vgae: bool = False,
        use_laplacian_pe: bool = False,
        pe_dim: int = 16,
    ):
        super().__init__()
        self.n_features = n_features
        self.aki_idx = aki_idx
        self.use_vgae = use_vgae
        self.use_laplacian_pe = use_laplacian_pe
        self.pe_dim = pe_dim

        # Feature identity embedding
        identity_dim = hidden_dim
        self.feature_embedding = nn.Embedding(n_features, identity_dim)

        # Laplacian PE projection (if enabled)
        pe_proj_dim = 0
        if use_laplacian_pe:
            pe_proj_dim = hidden_dim // 4
            self.pe_proj = nn.Linear(pe_dim, pe_proj_dim)
            self.register_buffer("_lap_pe", torch.zeros(n_features, pe_dim))

        # Input: [value, obs_mask] + identity + optional PE
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
                GraphBlock(encoder_type, current, od, oh, cc, dropout, edge_dim)
            )
            current = blocks[-1].actual_out
        self.encoder = nn.ModuleList(blocks)

        # VGAE heads (optional)
        if use_vgae:
            self.mu_head = nn.Linear(embedding_dim, embedding_dim)
            self.logvar_head = nn.Linear(embedding_dim, embedding_dim)

        # Decoder
        if decoder_type == "symmetric":
            self.decoder = SymmetricDecoder()
        elif decoder_type == "directed":
            self.decoder = DirectedDecoder(embedding_dim)
        elif decoder_type == "crosscorr":
            self.decoder = CrossCorrelationDecoder(embedding_dim)
        else:
            raise ValueError(f"Unknown decoder_type={decoder_type!r}")

        # AKI head: pooled mode
        self.aki_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def register_laplacian_pe(self, edge_index: torch.Tensor):
        """Precompute and cache LapPE from the graph topology."""
        pe = compute_laplacian_pe(edge_index, self.n_features, self.pe_dim)
        self._lap_pe = pe.to(self._lap_pe.device)

    def encode(self, x, edge_index, feature_id, edge_attr=None):
        parts = [x, self.feature_embedding(feature_id)]
        if self.use_laplacian_pe:
            pe = self._lap_pe[feature_id]  # [B*F, pe_dim]
            parts.append(self.pe_proj(pe))
        h = F.relu(self.input_proj(torch.cat(parts, dim=-1)))
        for block in self.encoder:
            h = block(h, edge_index, edge_attr)
        return h

    def reparameterise(self, z_flat: torch.Tensor):
        mu = self.mu_head(z_flat)
        logvar = self.logvar_head(z_flat)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl

    def forward(self, batch) -> ModelOutput:
        z_flat = self.encode(
            batch.x,
            batch.edge_index,
            batch.feature_id,
            batch.edge_attr,
        )

        # Optional VGAE
        kl_loss = None
        if self.use_vgae:
            z_flat, kl_loss = self.reparameterise(z_flat)

        z_by_graph = z_flat.view(batch.num_graphs, self.n_features, -1)

        # Association decoder
        association_hat = self.decoder(z_by_graph)

        # AKI prediction
        pooled = global_mean_pool(z_flat, batch.batch)
        logits = self.aki_head(pooled)

        return ModelOutput(
            z=z_flat,
            association_hat=association_hat,
            aki_logits=logits,
            kl_loss=kl_loss,
        )


# ── Loss functions ───────────────────────────────────────────


def association_loss(
    association_hat: torch.Tensor,
    association_target: torch.Tensor,
    ignore_diagonal: bool = True,
    directed: bool = False,
) -> torch.Tensor:
    """
    MSE between predicted and target association matrices.
    For directed=True, uses the full asymmetric matrix.
    For directed=False, symmetrises before comparison.
    """
    target = association_target.to(association_hat.device)
    target = target.unsqueeze(0).expand_as(association_hat)
    n = target.size(-1)

    if not directed:
        # Symmetrise both (for Spearman/MI targets)
        association_hat = (association_hat + association_hat.transpose(1, 2)) / 2
        target = (target + target.transpose(1, 2)) / 2

    if ignore_diagonal:
        mask = ~torch.eye(n, dtype=torch.bool, device=target.device)
        return F.mse_loss(association_hat[:, mask], target[:, mask])
    return F.mse_loss(association_hat, target)


# ── Builder ──────────────────────────────────────────────────


def build_model(
    n_features: int,
    aki_idx: int,
    hidden_dim: int = 64,
    embedding_dim: int = 16,
    n_layers: int = 2,
    encoder_type: str = "gcn",
    n_heads: int = 4,
    dropout: float = 0.1,
    n_classes: int = 2,
    edge_dim: int | None = None,
    decoder_type: str = "directed",
    use_vgae: bool = False,
    use_laplacian_pe: bool = False,
    pe_dim: int = 16,
    **_,
) -> FeatureGraphAutoencoderV2:
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
    )
