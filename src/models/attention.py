"""Attention-based fusion.

Only two "tokens" exist here (the pooled embedding of each protein -- no
per-residue embeddings are cached), so this is a single scaled dot-product
self-attention layer over the 2-token sequence [h_A, h_B], not a multi-layer
transformer -- an appropriately scoped amount of complexity for what's
actually available, not attention for its own sake.

Self-attention is permutation-EQUIVARIANT (attending token i to {A,B} gives
the same per-token output regardless of which position i sits in), but the
two attended outputs must still be combined symmetrically to get an
order-invariant prediction -- done here with a commutative sum,
same discipline as src/models/siamese.py.
"""
import torch
from torch import nn


class AttentionFusion(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 128):
        super().__init__()
        self.query = nn.Linear(input_dim, proj_dim)
        self.key = nn.Linear(input_dim, proj_dim)
        self.value = nn.Linear(input_dim, proj_dim)
        self.scale = proj_dim ** 0.5
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([h_a, h_b], dim=1)  # (batch, 2, input_dim)
        q, k, v = self.query(tokens), self.key(tokens), self.value(tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, 2, 2)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)  # (batch, 2, proj_dim)
        fused = attended.sum(dim=1)  # commutative pooling over the 2 tokens -> order-invariant
        return self.classifier(fused).squeeze(-1)
