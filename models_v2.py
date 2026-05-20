#!/usr/bin/env python3
"""
models_v2.py — Feature-graph autoencoder for ICI→AKI prediction.

Surgical fixes included:
  ① AKI node as query/target token — classify from z_aki or z_aki+context.
  ② Decoupled topology from association target — handled in train_v2.py.
  ③ Static association loss can be applied to feature identity embeddings.
  ④ WeightedDirGCNConv preserves edge weights while separating in/out flow.
  ⑤ Masked feature reconstruction is functional: model appends a pretext-mask channel
     and returns reconstruction predictions for masked nodes.
  ⑥ Directed / symmetric / cross-correlation decoders.
  ⑦ Optional VGAE and Laplacian positional encoding.

Readout modes:
  global_pool   : mean(all nodes)                  — legacy baseline
  non_aki_pool  : mean(non-AKI nodes)              — excludes masked outcome node
  aki_only      : z_aki                            — outcome-node query
  aki_concat    : concat(z_aki, mean(non-AKI))     — recommended default
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
    kl_loss: torch.Tensor | None = None
    recon_pred: torch.Tensor | None = (
        None  # [n_masked], logits/raw values for masked feature reconstruction
    )


# ─────────────────────────────────────────────────────────────
# Laplacian positional encoding
# ─────────────────────────────────────────────────────────────


def compute_laplacian_pe(
    edge_index: torch.Tensor, n_nodes: int, pe_dim: int
) -> torch.Tensor:
    """Precompute normalized-Laplacian eigenvector PE for a small graph."""
    ei = edge_index.detach().cpu()
    adj = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    adj[ei[0], ei[1]] = 1.0
    adj = (adj + adj.T).clamp(max=1.0)

    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    L_norm = torch.eye(n_nodes) - torch.diag(deg_inv_sqrt) @ adj @ torch.diag(
        deg_inv_sqrt
    )

    _, eigvecs = torch.linalg.eigh(L_norm)
    pe = eigvecs[:, 1 : pe_dim + 1]  # skip constant eigenvector
    if pe.shape[1] < pe_dim:
        pe = F.pad(pe, (0, pe_dim - pe.shape[1]))
    return pe.float()


# ─────────────────────────────────────────────────────────────
# Direction-aware weighted GCN
# ─────────────────────────────────────────────────────────────


class WeightedDirGCNConv(nn.Module):
    """
    Direction-aware GCN that preserves edge weights.

    For an edge_index representing source → target, `conv_in` aggregates along the
    original direction and `conv_out` aggregates along the reversed direction using
    the same edge weights. A root transform replaces self-loops.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_in = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
        self.conv_out = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
        self.root = nn.Linear(in_dim, out_dim)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ):
        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.view(-1)

        a = torch.sigmoid(self.alpha)
        x_in = self.conv_in(x, edge_index, edge_weight=edge_weight)
        x_out = self.conv_out(x, edge_index.flip(0), edge_weight=edge_weight)
        return a * x_in + (1.0 - a) * x_out + self.root(x)


# ─────────────────────────────────────────────────────────────
# Encoder blocks
# ─────────────────────────────────────────────────────────────


class GraphBlock(nn.Module):
    def __init__(
        self,
        conv_type: str,
        in_dim: int,
        out_dim: int,
        heads: int,
        concat: bool,
        dropout: float,
        edge_dim: int | None,
        use_dir_conv: bool = False,
    ):
        super().__init__()
        self.conv_type = conv_type.lower()
        self.use_dir_conv = use_dir_conv

        if use_dir_conv:
            # Direction-aware weighted GCN; intentionally GCN-based.
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ):
        if self.use_dir_conv:
            ew = edge_attr.squeeze(-1) if edge_attr is not None else None
            x = self.conv(x, edge_index, edge_weight=ew)
        elif self.conv_type == "gcn":
            ew = edge_attr.squeeze(-1) if edge_attr is not None else None
            x = self.conv(x, edge_index, edge_weight=ew)
        else:
            x = self.conv(x, edge_index, edge_attr=edge_attr)
        return F.dropout(F.elu(self.norm(x)), p=self.dropout, training=self.training)


# ─────────────────────────────────────────────────────────────
# Decoders
# ─────────────────────────────────────────────────────────────


class SymmetricDecoder(nn.Module):
    """Inner-product decoder: sigmoid(Z Zᵀ / sqrt(d))."""

    def forward(self, z_by_graph: torch.Tensor) -> torch.Tensor:
        d = z_by_graph.size(-1)
        return torch.sigmoid(z_by_graph @ z_by_graph.transpose(1, 2) / math.sqrt(d))


class DirectedDecoder(nn.Module):
    """Asymmetric decoder: A_hat[i,j] ≈ sigmoid(z_tgt_i · z_src_j / sqrt(d))."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.proj_src = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_tgt = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, z_by_graph: torch.Tensor) -> torch.Tensor:
        z_src = self.proj_src(z_by_graph)
        z_tgt = self.proj_tgt(z_by_graph)
        d = z_by_graph.size(-1)
        return torch.sigmoid(z_tgt @ z_src.transpose(1, 2) / math.sqrt(d))


class CrossCorrelationDecoder(nn.Module):
    """S2GAE-style pairwise element-product decoder."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, z_by_graph: torch.Tensor) -> torch.Tensor:
        B, F, d = z_by_graph.shape
        z_i = z_by_graph.unsqueeze(2).expand(B, F, F, d)
        z_j = z_by_graph.unsqueeze(1).expand(B, F, F, d)
        return torch.sigmoid(self.proj(z_i * z_j).squeeze(-1))


# ─────────────────────────────────────────────────────────────
# Masked feature reconstruction head
# ─────────────────────────────────────────────────────────────


class MaskedFeatureReconHead(nn.Module):
    """Predict a masked feature value from the masked node embedding."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_masked_nodes: torch.Tensor) -> torch.Tensor:
        return self.head(z_masked_nodes).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────


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
        if readout not in self.READOUT_MODES:
            raise ValueError(
                f"readout must be one of {self.READOUT_MODES}, got {readout!r}"
            )
        if aki_idx is None and readout in ("non_aki_pool", "aki_only", "aki_concat"):
            raise ValueError("aki_idx is required for AKI-aware readout modes")

        self.n_features = n_features
        self.aki_idx = aki_idx
        self.readout = readout
        self.use_vgae = use_vgae
        self.use_laplacian_pe = use_laplacian_pe
        self.use_masked_recon = use_masked_recon
        self.pe_dim = pe_dim

        # If masked reconstruction is enabled, encode appends a third channel:
        # [value, observed_mask, pretext_mask].
        expected_node_input_dim = 3 if use_masked_recon else node_input_dim
        self.node_input_dim = expected_node_input_dim

        identity_dim = hidden_dim
        self.feature_embedding = nn.Embedding(n_features, identity_dim)

        pe_proj_dim = 0
        if use_laplacian_pe:
            pe_proj_dim = hidden_dim // 4
            self.pe_proj = nn.Linear(pe_dim, pe_proj_dim)
            self.register_buffer("_lap_pe", torch.zeros(n_features, pe_dim))

        self.input_proj = nn.Linear(
            expected_node_input_dim + identity_dim + pe_proj_dim, hidden_dim
        )

        blocks: list[nn.Module] = []
        current = hidden_dim
        for i in range(n_layers):
            is_last = i == n_layers - 1
            od = embedding_dim if is_last else hidden_dim
            oh = 1 if is_last else n_heads
            cc = not is_last
            block = GraphBlock(
                encoder_type,
                current,
                od,
                oh,
                cc,
                dropout,
                edge_dim,
                use_dir_conv=use_dir_conv,
            )
            blocks.append(block)
            current = block.actual_out
        self.encoder = nn.ModuleList(blocks)

        if use_vgae:
            self.mu_head = nn.Linear(embedding_dim, embedding_dim)
            self.logvar_head = nn.Linear(embedding_dim, embedding_dim)

        if decoder_type == "symmetric":
            self.decoder = SymmetricDecoder()
        elif decoder_type == "directed":
            self.decoder = DirectedDecoder(embedding_dim)
        elif decoder_type == "crosscorr":
            self.decoder = CrossCorrelationDecoder(embedding_dim)
        else:
            raise ValueError(f"Unknown decoder_type={decoder_type!r}")

        # Static association uses feature identity embeddings projected into the
        # same latent space as patient-specific z, then reuses the chosen decoder.
        self.static_assoc_proj = nn.Linear(identity_dim, embedding_dim)

        aki_in = (
            embedding_dim
            if readout in ("global_pool", "non_aki_pool", "aki_only")
            else 2 * embedding_dim
        )
        self.aki_head = nn.Sequential(
            nn.Linear(aki_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        if use_masked_recon:
            self.recon_head = MaskedFeatureReconHead(embedding_dim, hidden_dim)

    def register_laplacian_pe(self, edge_index: torch.Tensor):
        pe = compute_laplacian_pe(edge_index, self.n_features, self.pe_dim)
        self._lap_pe = pe.to(self._lap_pe.device)

    def _augment_node_input(
        self, x: torch.Tensor, pretext_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Append pretext-mask channel when masked reconstruction is enabled."""
        if not self.use_masked_recon:
            return x
        if x.size(-1) == self.node_input_dim:
            return x
        if x.size(-1) != 2:
            raise ValueError(
                f"Expected node input with 2 or {self.node_input_dim} channels, got {x.size(-1)}"
            )
        if pretext_mask is None:
            pm = torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device)
        else:
            pm = pretext_mask.to(device=x.device, dtype=x.dtype).view(-1, 1)
        return torch.cat([x, pm], dim=-1)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        feature_id: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        pretext_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._augment_node_input(x, pretext_mask)
        parts = [x, self.feature_embedding(feature_id)]
        if self.use_laplacian_pe:
            parts.append(self.pe_proj(self._lap_pe[feature_id]))
        h = F.relu(self.input_proj(torch.cat(parts, dim=-1)))
        for block in self.encoder:
            h = block(h, edge_index, edge_attr)
        return h

    def reparameterise(self, z_flat: torch.Tensor):
        mu = self.mu_head(z_flat)
        logvar = self.logvar_head(z_flat).clamp(min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return z, kl

    def _readout(self, z_by_graph: torch.Tensor) -> torch.Tensor:
        if self.readout == "global_pool":
            return z_by_graph.mean(dim=1)

        mask = torch.ones(self.n_features, dtype=torch.bool, device=z_by_graph.device)
        mask[self.aki_idx] = False

        if self.readout == "non_aki_pool":
            return z_by_graph[:, mask, :].mean(dim=1)
        if self.readout == "aki_only":
            return z_by_graph[:, self.aki_idx, :]
        if self.readout == "aki_concat":
            z_aki = z_by_graph[:, self.aki_idx, :]
            z_ctx = z_by_graph[:, mask, :].mean(dim=1)
            return torch.cat([z_aki, z_ctx], dim=-1)
        raise RuntimeError(f"Unhandled readout={self.readout!r}")

    def static_association(self) -> torch.Tensor:
        """Patient-invariant association from feature identity embeddings."""
        z_static = self.static_assoc_proj(self.feature_embedding.weight).unsqueeze(
            0
        )  # [1, F, emb]
        return self.decoder(z_static)

    def forward(self, batch, pretext_mask: torch.Tensor | None = None) -> ModelOutput:
        z_flat = self.encode(
            batch.x,
            batch.edge_index,
            batch.feature_id,
            getattr(batch, "edge_attr", None),
            pretext_mask=pretext_mask,
        )

        kl_loss = None
        if self.use_vgae:
            z_flat, kl_loss = self.reparameterise(z_flat)

        z_by_graph = z_flat.reshape(batch.num_graphs, self.n_features, -1)
        association_hat = self.decoder(z_by_graph)
        cls_features = self._readout(z_by_graph)
        logits = self.aki_head(cls_features)

        recon_pred = None
        if self.use_masked_recon and pretext_mask is not None and pretext_mask.any():
            recon_pred = self.recon_head(z_flat[pretext_mask])

        return ModelOutput(
            z=z_flat,
            association_hat=association_hat,
            aki_logits=logits,
            kl_loss=kl_loss,
            recon_pred=recon_pred,
        )


# ─────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────


def association_mse(
    a_hat: torch.Tensor,
    a_target: torch.Tensor,
    ignore_diag: bool = True,
    directed: bool = False,
) -> torch.Tensor:
    """MSE between predicted and target association matrices."""
    target = a_target.to(a_hat.device)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    target = target.expand_as(a_hat)

    if not directed:
        a_hat = (a_hat + a_hat.transpose(1, 2)) / 2.0
        target = (target + target.transpose(1, 2)) / 2.0

    n = target.size(-1)
    if ignore_diag:
        mask = ~torch.eye(n, dtype=torch.bool, device=target.device)
        return F.mse_loss(a_hat[:, mask], target[:, mask])
    return F.mse_loss(a_hat, target)


def static_association_mse(
    model: FeatureGraphAutoencoderV2,
    a_target: torch.Tensor,
    ignore_diag: bool = True,
    directed: bool = False,
) -> torch.Tensor:
    """Association loss on feature identity embeddings, not patient-specific z."""
    a_hat = model.static_association()  # [1, F, F]
    return association_mse(a_hat, a_target, ignore_diag=ignore_diag, directed=directed)


# ─────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────


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
    decoder_type: str = "symmetric",
    use_vgae: bool = False,
    use_laplacian_pe: bool = False,
    pe_dim: int = 16,
    use_dir_conv: bool = False,
    readout: str = "aki_concat",
    use_masked_recon: bool = False,
    **_,
) -> FeatureGraphAutoencoderV2:
    return FeatureGraphAutoencoderV2(
        n_features=n_features,
        node_input_dim=3 if use_masked_recon else 2,
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
