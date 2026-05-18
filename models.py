#!/usr/bin/env python3
"""
models.py — Feature-graph autoencoder for ICI-AKI prediction.

Core idea
---------
X is an [N, K] patient-feature matrix.  The graph G is over K feature nodes.
Each patient row x_i is a graph signal on the feature network.  The model
reconstructs X and predicts AKI:

    L = lambda_rec * L_rec(X_hat, X) + L_AKI(P_AKI, y_AKI)

AKI prediction is the AKI node's reconstruction: sigma(X_hat[:, aki_idx]).
No separate classification head — the decoder must learn AKI through the
graph structure.  A --separate_aki_head ablation flag exists for comparison.

Inductive over patients: new rows can be scored without retraining.

Architecture ref: SiGra (Song et al. 2023, Nat Comms), GATE (Salehi 2019).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, TransformerConv

# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────


@dataclass
class ModelOutput:
    x_hat: torch.Tensor  # [B, F], reconstruction values (all nodes)
    aki_logits: torch.Tensor  # [B], AKI logits (from recon or separate head)
    z: torch.Tensor  # [B, F, latent]


def _repeat_graph(
    edge_index: torch.Tensor,
    num_graphs: int,
    num_nodes: int,
) -> torch.Tensor:
    """Repeat one feature graph into B disconnected patient copies."""
    if num_graphs == 1:
        return edge_index
    edge_index = edge_index.long()
    num_edges = edge_index.size(1)
    offsets = (
        torch.arange(num_graphs, device=edge_index.device)
        .repeat_interleave(num_edges)
        .mul(num_nodes)
    )
    return edge_index.repeat(1, num_graphs) + offsets.unsqueeze(0)


def _repeat_edge_attr(
    edge_attr: Optional[torch.Tensor], num_graphs: int
) -> Optional[torch.Tensor]:
    if edge_attr is None:
        return None
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)
    return edge_attr.repeat(num_graphs, 1)


# ──────────────────────────────────────────────────────────────
#  Graph conv block
# ──────────────────────────────────────────────────────────────


class GraphBlock(nn.Module):
    """GATv2 or TransformerConv block with LayerNorm + ELU + dropout."""

    def __init__(
        self,
        conv_type: str,
        in_dim: int,
        out_dim: int,
        heads: int,
        concat: bool,
        dropout: float,
        edge_dim: Optional[int],
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        conv_type = conv_type.lower()
        if conv_type == "gatv2":
            self.conv = GATv2Conv(
                in_dim,
                out_dim,
                heads=heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=add_self_loops,
                fill_value=1.0,  # self-loop edge weight
                residual=True,
            )
        elif conv_type == "transformer":
            self.conv = TransformerConv(
                in_dim,
                out_dim,
                heads=heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                beta=True,
            )
        else:
            raise ValueError(f"Unknown conv_type={conv_type!r}")

        actual_out = out_dim * heads if concat else out_dim
        self.norm = nn.LayerNorm(actual_out)
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        x = F.elu(x)
        return F.dropout(x, p=self.dropout, training=self.training)


# ──────────────────────────────────────────────────────────────
#  Main model
# ──────────────────────────────────────────────────────────────


class FeatureGraphAE(nn.Module):
    """
    Patient-as-graph-signal feature-graph autoencoder.

    Default: AKI prediction = sigma(x_hat[:, aki_idx])  (reconstruction).
    Ablation: AKI prediction = MLP(z_aki, z_pool)       (separate head).

    Parameters
    ----------
    num_features : K feature nodes including AKI
    aki_idx      : column index of AKI in the feature matrix
    layers       : GNN depth, 2 or 3
    use_recon_aki: True = AKI logit from decoder reconstruction (whiteboard)
                   False = AKI logit from separate MLP head (ablation)
    """

    def __init__(
        self,
        num_features: int,
        aki_idx: int,
        node_emb_dim: int = 8,
        hidden: int = 64,
        latent: int = 32,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.15,
        edge_dim: Optional[int] = 1,
        conv_type: str = "gatv2",
        use_recon_aki: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if layers not in (2, 3):
            raise ValueError("layers must be 2 or 3")

        self.num_features = int(num_features)
        self.aki_idx = int(aki_idx)
        self.use_recon_aki = use_recon_aki
        self.node_emb = nn.Embedding(num_features, node_emb_dim)

        # Node input = value(1) + missing_flag(1) + aki_flag(1) + emb
        in_dim = 1 + 1 + 1 + node_emb_dim

        # ── Encoder ──────────────────────────────────────────────
        enc_blocks = []
        current = in_dim
        for layer_idx in range(layers):
            is_last = layer_idx == layers - 1
            out_dim = latent if is_last else hidden
            out_heads = 1 if is_last else heads
            concat = not is_last
            enc_blocks.append(
                GraphBlock(
                    conv_type=conv_type,
                    in_dim=current,
                    out_dim=out_dim,
                    heads=out_heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=add_self_loops,
                )
            )
            current = out_dim * out_heads if concat else out_dim
        self.encoder = nn.ModuleList(enc_blocks)

        # ── GNN Decoder (SiGra-style) ────────────────────────────
        self.dec1 = GraphBlock(
            conv_type=conv_type,
            in_dim=latent,
            out_dim=hidden,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=add_self_loops,
        )
        self.dec2 = GraphBlock(
            conv_type=conv_type,
            in_dim=hidden * heads,
            out_dim=hidden,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=add_self_loops,
        )
        self.recon_head = nn.Linear(hidden, 1)

        # ── Separate AKI head (ablation only) ────────────────────
        self.aki_head = nn.Sequential(
            nn.Linear(latent * 2, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _make_node_input(
        self,
        x: torch.Tensor,
        missing_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """x: [B, F] → [B*F, in_dim]"""
        batch_size, num_features = x.shape
        device = x.device
        node_ids = torch.arange(num_features, device=device).repeat(batch_size)
        values = x.reshape(batch_size * num_features, 1)

        if missing_mask is None:
            miss = torch.zeros_like(values)
        else:
            miss = missing_mask.reshape(batch_size * num_features, 1).float()

        is_aki = (node_ids == self.aki_idx).float().unsqueeze(-1)
        emb = self.node_emb(node_ids)
        return torch.cat([values, miss, is_aki, emb], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        batch_size, num_features = x.shape
        edge_index_b = _repeat_graph(edge_index.to(x.device), batch_size, num_features)
        edge_attr_b = _repeat_edge_attr(
            edge_attr.to(x.device) if edge_attr is not None else None,
            batch_size,
        )

        # Encode
        h = self._make_node_input(x, missing_mask)
        for block in self.encoder:
            h = block(h, edge_index_b, edge_attr_b)

        z = h.view(batch_size, num_features, -1)  # [B, F, latent]

        # Decode (GNN decoder → scalar per node)
        d = self.dec1(h, edge_index_b, edge_attr_b)
        d = self.dec2(d, edge_index_b, edge_attr_b)
        x_hat = self.recon_head(d).view(batch_size, num_features)

        # ── AKI logit ────────────────────────────────────────────
        if self.use_recon_aki:
            # Whiteboard design: AKI prediction IS the reconstruction
            aki_logits = x_hat[:, self.aki_idx]
        else:
            # Ablation: separate MLP head
            z_aki = z[:, self.aki_idx, :]
            z_pool = z.mean(dim=1)
            aki_logits = self.aki_head(torch.cat([z_aki, z_pool], dim=-1)).squeeze(-1)

        return ModelOutput(x_hat=x_hat, aki_logits=aki_logits, z=z)

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(
            self.forward(x, edge_index, edge_attr, missing_mask).aki_logits
        )


def build_model(
    num_features: int = 38,
    aki_idx: int = 37,
    node_emb_dim: int = 8,
    hidden: int = 64,
    latent: int = 32,
    heads: int = 4,
    layers: int = 3,
    dropout: float = 0.15,
    edge_dim: Optional[int] = 1,
    conv_type: str = "gatv2",
    use_recon_aki: bool = True,
    add_self_loops: bool = True,
    **_: Dict,
) -> FeatureGraphAE:
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
        use_recon_aki=use_recon_aki,
        add_self_loops=add_self_loops,
    )
