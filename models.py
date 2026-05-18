#!/usr/bin/env python3
"""
models.py — Feature-graph risk model.

Architecture (per GPT diagnosis):
  Encoder: GATv2 on feature graph → z [F, latent] (node embeddings)
  Risk decoder: β_i = MLP(z_i, z_AKI) → logit_p = b + Σ β_i * x_ip
  Skip: raw logistic regression β_skip·x (guarantees ≥ LogReg baseline)

Key insight: the decoder must produce PATIENT-LEVEL logits via a SHARED
classifier (β·x), NOT via Linear(latent, N_patients). The old decoder
had one output neuron per patient ID — test patients got zero gradient.

The learned β_i are directly interpretable alongside the attention heatmap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch_geometric.nn import GATv2Conv

# ══════════════════════════════════════════════════════════════
# Risk decoder: graph embeddings → per-feature weights → patient logits
# ══════════════════════════════════════════════════════════════


class FeatureGraphRiskDecoder(nn.Module):
    """
    Shared risk decoder.

    For each non-AKI feature i:
      β_i = MLP(concat(z_i, z_AKI))    ← weight from graph embedding
    For each patient p:
      logit_p = bias + Σ_i β_i * x_ip  ← weighted sum of feature values

    This restores patient exchangeability: logit depends on patient's
    FEATURE VALUES, not their column index in the tensor.
    """

    def __init__(self, latent, hidden=32):
        super().__init__()
        self.beta_mlp = nn.Sequential(
            nn.Linear(2 * latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, z, x, aki_idx):
        """
        Args:
            z: [F, latent] node embeddings from encoder
            x: [F, N] feature matrix (with AKI row)
            aki_idx: int
        Returns:
            logits: [N] patient-level AKI logits
            beta: [F-1] learned per-feature weights
        """
        F_nodes = z.size(0)
        feat_idx = [i for i in range(F_nodes) if i != aki_idx]

        z_feat = z[feat_idx]  # [F-1, latent]
        z_aki = z[aki_idx].unsqueeze(0).expand_as(z_feat)  # [F-1, latent]

        # Per-feature weight from graph embeddings
        beta = self.beta_mlp(torch.cat([z_feat, z_aki], dim=1)).squeeze(-1)  # [F-1]

        # Patient logits = weighted sum of features
        x_feat = x[feat_idx, :]  # [F-1, N]
        logits = self.bias + beta @ x_feat  # [N]

        return logits, beta


# ══════════════════════════════════════════════════════════════
# Encoder: GATv2 on feature graph
# ══════════════════════════════════════════════════════════════


class FeatureGraphRiskModel(nn.Module):
    """
    Feature-graph risk model with LogReg skip connection.

    logit_p = raw_skip(x_p) + GAT_risk(z, x_p)

    The raw_skip ensures the model can match LogReg baseline immediately.
    The GAT path learns graph-structured residual signal.
    """

    def __init__(
        self,
        num_features,
        aki_idx,
        hidden=32,
        latent=16,
        heads=2,
        dropout=0.1,
        edge_dim=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.aki_idx = aki_idx
        self.non_aki = [i for i in range(num_features) if i != aki_idx]

        # ── GATv2 encoder ─────────────────────────────────────
        # in_channels = N (patient values per feature node)
        # Set dynamically in forward() via lazy init or pass explicitly
        self.conv1 = None  # lazy init on first forward
        self.conv2 = None
        self.norm1 = None
        self.hidden = hidden
        self.latent = latent
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._encoder_built = False

        # ── Risk decoder ──────────────────────────────────────
        self.risk_decoder = FeatureGraphRiskDecoder(latent, hidden)

        # ── Raw LogReg skip (no graph needed) ─────────────────
        self.raw_skip = nn.Linear(num_features - 1, 1, bias=True)

        # ── Optional: masked feature reconstruction head ──────
        self.recon_head = nn.Linear(latent, 1)

    def _build_encoder(self, in_channels):
        """Lazy encoder init — called on first forward with actual N."""
        device = self.raw_skip.weight.device
        self.conv1 = GATv2Conv(
            in_channels,
            self.hidden,
            heads=self.heads,
            concat=True,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
            add_self_loops=False,
            residual=True,
        ).to(device)
        self.norm1 = nn.LayerNorm(self.hidden * self.heads).to(device)
        self.conv2 = GATv2Conv(
            self.hidden * self.heads,
            self.latent,
            heads=1,
            concat=False,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
            add_self_loops=False,
            residual=True,
        ).to(device)
        self._encoder_built = True

    def encode(self, x, edge_index, edge_attr=None):
        """x: [F, N] → z: [F, latent]"""
        if not self._encoder_built:
            self._build_encoder(x.size(1))
        ea = edge_attr if self.edge_dim else None
        h = self.conv1(x, edge_index, edge_attr=ea)
        h = self.norm1(h)
        h = Fnn.elu(h)
        h = Fnn.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index, edge_attr=ea)
        return z

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [F, N] feature matrix (AKI row present but zeroed externally)
            edge_index: [2, E]
            edge_attr: [E, 1] optional edge weights
        Returns:
            logits: [N] patient-level AKI logits
            beta: [F-1] learned feature weights from graph
            recon: [F, N] optional reconstruction for masked feature modeling
        """
        # Encode
        z = self.encode(x, edge_index, edge_attr)  # [F, latent]

        # Risk decoder: graph-derived feature weights
        gat_logits, beta = self.risk_decoder(z, x, self.aki_idx)  # [N], [F-1]

        # Raw LogReg skip: direct path from features to prediction
        x_non_aki = x[self.non_aki, :].T  # [N, F-1]
        skip_logits = self.raw_skip(x_non_aki).squeeze(-1)  # [N]

        # Combined logits
        logits = skip_logits + gat_logits  # [N]

        # Optional reconstruction for masked feature modeling
        recon = self.recon_head(z).squeeze(-1)  # [F]... but we need [F, N]
        # Actually: recon per node is z[i] → scalar. For masked feature modeling,
        # we predict masked feature values. This is handled in the loss function.

        return logits, beta, z

    def get_attention(self, x, edge_index, edge_attr=None):
        """Extract GATv2 attention weights for heatmap."""
        self.eval()
        if not self._encoder_built:
            self._build_encoder(x.size(1))
        ea = edge_attr if self.edge_dim else None
        with torch.no_grad():
            h1, (ei1, aw1) = self.conv1(
                x,
                edge_index,
                edge_attr=ea,
                return_attention_weights=True,
            )
            h1 = self.norm1(h1)
            h1 = Fnn.elu(h1)
            _, (ei2, aw2) = self.conv2(
                h1,
                edge_index,
                edge_attr=ea,
                return_attention_weights=True,
            )
        return {
            "layer1": {"edge_index": ei1.cpu(), "attention": aw1.cpu()},
            "layer2": {"edge_index": ei2.cpu(), "attention": aw2.cpu()},
        }


def build_model(
    num_features=38,
    aki_idx=37,
    hidden=32,
    latent=16,
    heads=2,
    dropout=0.1,
    edge_dim=None,
    **kwargs,
):
    """Factory function."""
    return FeatureGraphRiskModel(
        num_features=num_features,
        aki_idx=aki_idx,
        hidden=hidden,
        latent=latent,
        heads=heads,
        dropout=dropout,
        edge_dim=edge_dim,
    )
