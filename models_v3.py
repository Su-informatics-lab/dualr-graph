#!/usr/bin/env python3
"""
models_v3.py — Su's feature-graph autoencoder for ICI→AKI prediction.

Key changes from v2:
  ① DirectedWeightedConv: implements Su's message passing equation exactly:
       v_i^(l+1) = W_self · v_i^(l) + Σ_{j∈N(i)} W_neigh · v_j^(l) · P(v_i|v_j)
     No symmetrization. Direction and LLM edge weights preserved.

  ② Feature value decoder: reconstructs X̂ ∈ R^{38} per patient from z,
     NOT a 38×38 association matrix. Each node embedding z_j → x̂_j (scalar).

  ③ AKI from reconstruction: x̂_aki → sigmoid → P(AKI). No separate MLP head.
     The AKI prediction is a byproduct of feature reconstruction.

  ④ Loss = λ_rec · L_rec + L_aki  (+ optional λ_assoc · L_assoc for regularization)
     L_rec  = per-feature reconstruction (MSE for continuous, BCE for binary)
     L_aki  = BCE(sigmoid(x̂_aki), AKI_label)  +  MSE(sigmoid(x̂_aki), AKI_label)
     L_assoc = optional auxiliary: MSE(sigmoid(z@z.T/√d), |Pearson corr|)

  ⑤ obs_mask as second input channel: [value, obs_mask] per node.
     AKI is always [0, 0]. Missing features are [imputed_value, 0].
     Model learns to reconstruct masked values from neighbor messages.

Architecture (matching Su's sketches):
  Input: [value, obs_mask] + feature identity embedding
  → Linear projection → hidden_dim
  → L directed GCN layers (LayerNorm + ELU, dropout)
  → per-node embeddings z ∈ R^{38 × emb_dim}
  → shared decoder MLP: z_j → x̂_j
  → split loss: reconstruction (non-AKI) + AKI prediction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# ─────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────


@dataclass
class ModelOutput:
    z: torch.Tensor  # [B*F, emb] flat node embeddings
    x_hat: torch.Tensor  # [B, F] reconstructed feature values (raw, pre-sigmoid)
    aki_prob: torch.Tensor  # [B] sigmoid(x_hat[:, aki_idx])
    association_hat: torch.Tensor | None = None  # [B, F, F] optional auxiliary


# ─────────────────────────────────────────────────────────────
# Directed weighted message passing (Su's equation)
# ─────────────────────────────────────────────────────────────


class DirectedWeightedConv(MessagePassing):
    """
    Su's directed message passing:

        v_i^(l+1) = W_self · v_i^(l)
                   + Σ_{j ∈ N(i)} W_neigh · v_j^(l) · P(v_i | v_j)

    edge_index: [2, E] where edge (j → i) means j is source, i is target.
    edge_weight: [E] = P(v_i | v_j) for each edge.

    No symmetrization. No self-loop injection. Direction fully preserved.
    """

    def __init__(self, in_dim: int, out_dim: int):
        # aggr='add' sums messages; flow='source_to_target' means
        # message goes from edge_index[0] to edge_index[1].
        super().__init__(aggr="add", flow="source_to_target")
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-loop transform (independent of graph structure).
        out_self = self.W_self(x)

        # Neighbor aggregation through directed edges.
        out_neigh = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out_self + out_neigh

    def message(
        self, x_j: torch.Tensor, edge_weight: torch.Tensor | None
    ) -> torch.Tensor:
        # x_j: source node features [E, in_dim]
        # edge_weight: P(v_i | v_j) [E]
        msg = self.W_neigh(x_j)
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)
        return msg


# ─────────────────────────────────────────────────────────────
# Encoder block
# ─────────────────────────────────────────────────────────────


class DirectedGCNBlock(nn.Module):
    """One layer: DirectedWeightedConv → LayerNorm → ELU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = DirectedWeightedConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_weight)
        h = F.elu(self.norm(h))
        return F.dropout(h, p=self.dropout, training=self.training)


# ─────────────────────────────────────────────────────────────
# Feature value decoder
# ─────────────────────────────────────────────────────────────


class FeatureValueDecoder(nn.Module):
    """
    Decode per-node embedding z_j ∈ R^emb → scalar x̂_j.

    Shared MLP across all nodes. The feature identity is already
    encoded in z_j by the encoder's identity embedding, so a shared
    decoder can distinguish which feature it's reconstructing.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [N_total, emb] → [N_total, 1] → squeeze to [N_total]."""
        return self.mlp(z).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Optional symmetric association decoder (auxiliary regularizer)
# ─────────────────────────────────────────────────────────────


class SymmetricAssociationDecoder(nn.Module):
    """sigmoid(z @ z.T / √d) — kept as optional auxiliary objective."""

    def forward(self, z_by_graph: torch.Tensor) -> torch.Tensor:
        d = z_by_graph.size(-1)
        return torch.sigmoid(z_by_graph @ z_by_graph.transpose(1, 2) / math.sqrt(d))


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────


class FeatureGraphAutoencoderV3(nn.Module):
    """
    Su's feature-graph autoencoder.

    Encode:  X (patient features on directed LLM graph) → z (per-node embeddings)
    Decode:  z → X̂ (reconstructed feature values, including AKI)
    Predict: P(AKI) = sigmoid(X̂_aki)
    """

    def __init__(
        self,
        n_features: int,
        aki_idx: int,
        hidden_dim: int = 64,
        embedding_dim: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_assoc_aux: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.aki_idx = aki_idx
        self.embedding_dim = embedding_dim
        self.use_assoc_aux = use_assoc_aux

        # ── Input projection ──
        # node_input = [value, obs_mask] (2 channels) + identity embedding
        self.feature_embedding = nn.Embedding(n_features, hidden_dim)
        self.input_proj = nn.Linear(2 + hidden_dim, hidden_dim)

        # ── Directed GCN encoder ──
        blocks = []
        for i in range(n_layers):
            in_d = (
                hidden_dim
                if i == 0
                else (embedding_dim if i == n_layers - 1 else hidden_dim)
            )
            # First layer: hidden→hidden, last layer: hidden→embedding_dim
            if i == 0:
                in_d = hidden_dim
                out_d = hidden_dim if n_layers > 1 else embedding_dim
            elif i == n_layers - 1:
                in_d = hidden_dim
                out_d = embedding_dim
            else:
                in_d = hidden_dim
                out_d = hidden_dim
            blocks.append(DirectedGCNBlock(in_d, out_d, dropout))
        self.encoder = nn.ModuleList(blocks)

        # ── Feature value decoder ──
        self.value_decoder = FeatureValueDecoder(embedding_dim, hidden_dim)

        # ── Optional association decoder (auxiliary regularizer) ──
        if use_assoc_aux:
            self.assoc_decoder = SymmetricAssociationDecoder()

    def encode(
        self,
        x: torch.Tensor,  # [B*F, 2] = [value, obs_mask]
        edge_index: torch.Tensor,  # [2, E]
        feature_id: torch.Tensor,  # [B*F] feature indices 0..37
        edge_weight: torch.Tensor | None = None,  # [E] = P(v_i|v_j)
    ) -> torch.Tensor:
        """Encode → per-node embeddings z ∈ R^{B*F × emb_dim}."""
        # Concatenate node input with feature identity embedding.
        identity = self.feature_embedding(feature_id)  # [B*F, hidden]
        h = torch.cat([x, identity], dim=-1)  # [B*F, 2 + hidden]
        h = F.relu(self.input_proj(h))  # [B*F, hidden]

        # L layers of directed message passing.
        for block in self.encoder:
            h = block(h, edge_index, edge_weight)

        return h  # [B*F, emb_dim]

    def decode(self, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Decode per-node embeddings to reconstructed feature values.

        z: [B*F, emb_dim] → x_hat: [B, F] (raw logits, not sigmoid'd).
        """
        x_hat_flat = self.value_decoder(z)  # [B*F]
        return x_hat_flat.view(batch_size, self.n_features)  # [B, F]

    def forward(self, batch) -> ModelOutput:
        """
        Full forward: encode → decode → reconstruct.

        batch: PyG Batch with batch.x, batch.edge_index, batch.feature_id,
               and optionally batch.edge_weight (LLM P(v_i|v_j)).
        """
        edge_weight = getattr(batch, "edge_weight", None)

        z = self.encode(
            batch.x,
            batch.edge_index,
            batch.feature_id,
            edge_weight=edge_weight,
        )

        # Feature value reconstruction.
        x_hat = self.decode(z, batch.num_graphs)  # [B, F]

        # AKI probability from reconstruction.
        aki_prob = torch.sigmoid(x_hat[:, self.aki_idx])  # [B]

        # Optional association matrix (auxiliary).
        assoc = None
        if self.use_assoc_aux:
            z_by_graph = z.view(batch.num_graphs, self.n_features, -1)
            assoc = self.assoc_decoder(z_by_graph)

        return ModelOutput(
            z=z,
            x_hat=x_hat,
            aki_prob=aki_prob,
            association_hat=assoc,
        )


# ─────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────


def reconstruction_loss(
    x_hat: torch.Tensor,  # [B, F] raw decoder output
    x_true: torch.Tensor,  # [B, F] true feature values
    obs_mask: torch.Tensor,  # [B, F] 1=observed, 0=missing/masked
    binary_mask: torch.Tensor,  # [F] 1=binary feature, 0=continuous
    aki_idx: int,
) -> torch.Tensor:
    """
    Per-feature reconstruction loss, excluding AKI.

    Only computes loss on OBSERVED features (obs_mask=1).
    Binary features: BCE.  Continuous features: SmoothL1.
    AKI is excluded — it has its own loss.
    """
    B, n_f = x_hat.shape
    binary = binary_mask.bool().to(x_hat.device)

    # Build mask: observed AND not AKI.
    feat_mask = obs_mask.clone()  # [B, F]
    feat_mask[:, aki_idx] = 0.0  # exclude AKI

    losses = []
    weights = []

    # Binary features (excluding AKI).
    bin_cols = binary.clone()
    bin_cols[aki_idx] = False
    bin_eligible = feat_mask[:, bin_cols]  # [B, n_bin]
    if bin_eligible.sum() > 0:
        pred_bin = x_hat[:, bin_cols]
        true_bin = x_true[:, bin_cols].clamp(0, 1)
        # Masked mean: only observed entries.
        bce = F.binary_cross_entropy_with_logits(pred_bin, true_bin, reduction="none")
        bin_loss = (bce * bin_eligible).sum() / bin_eligible.sum().clamp(min=1)
        losses.append(bin_loss)
        weights.append(bin_eligible.sum().item())

    # Continuous features.
    cont_cols = ~binary
    cont_eligible = feat_mask[:, cont_cols]
    if cont_eligible.sum() > 0:
        pred_cont = x_hat[:, cont_cols]
        true_cont = x_true[:, cont_cols]
        sl1 = F.smooth_l1_loss(pred_cont, true_cont, reduction="none")
        cont_loss = (sl1 * cont_eligible).sum() / cont_eligible.sum().clamp(min=1)
        losses.append(cont_loss)
        weights.append(cont_eligible.sum().item())

    if not losses:
        return torch.tensor(0.0, device=x_hat.device, requires_grad=True)

    # Weighted average of binary and continuous losses.
    w = torch.tensor(weights, device=x_hat.device)
    l = torch.stack(losses)
    return (l * w).sum() / w.sum().clamp(min=1)


def aki_loss(
    aki_prob: torch.Tensor,  # [B] sigmoid output
    aki_true: torch.Tensor,  # [B] 0 or 1
) -> torch.Tensor:
    """
    AKI prediction loss: BCE + MSE (Su's sketch shows both).

    BCE gives good gradient for classification.
    MSE pushes probability closer to 0/1, acts as calibration.
    """
    aki_true_f = aki_true.float()
    # BCE on probabilities (numerically stable version via logits would be
    # better, but aki_prob is already sigmoided from x_hat).
    bce = F.binary_cross_entropy(aki_prob, aki_true_f, reduction="mean")
    mse = F.mse_loss(aki_prob, aki_true_f, reduction="mean")
    return bce + mse


def association_aux_loss(
    association_hat: torch.Tensor,  # [B, F, F]
    association_target: torch.Tensor,  # [F, F] train-set |Pearson corr|
    ignore_diag: bool = True,
) -> torch.Tensor:
    """Optional auxiliary: regularize z@z.T to match train-set correlations."""
    if association_hat is None:
        return torch.tensor(0.0, device=association_target.device, requires_grad=True)

    target = association_target.unsqueeze(0).expand_as(association_hat)
    F_dim = target.size(-1)

    if ignore_diag:
        mask = ~torch.eye(F_dim, dtype=torch.bool, device=target.device)
        return F.mse_loss(association_hat[:, mask], target[:, mask])
    return F.mse_loss(association_hat, target)


def total_loss(
    out: ModelOutput,
    x_true: torch.Tensor,  # [B, F]
    obs_mask: torch.Tensor,  # [B, F]
    binary_mask: torch.Tensor,  # [F]
    aki_true: torch.Tensor,  # [B]
    aki_idx: int,
    assoc_target: torch.Tensor | None = None,
    lambda_rec: float = 1.0,
    lambda_aki: float = 1.0,
    lambda_assoc: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute all losses and return a dict for logging."""
    l_rec = reconstruction_loss(out.x_hat, x_true, obs_mask, binary_mask, aki_idx)
    l_aki = aki_loss(out.aki_prob, aki_true)

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
    dropout: float = 0.1,
    use_assoc_aux: bool = True,
) -> FeatureGraphAutoencoderV3:
    return FeatureGraphAutoencoderV3(
        n_features=n_features,
        aki_idx=aki_idx,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        dropout=dropout,
        use_assoc_aux=use_assoc_aux,
    )
