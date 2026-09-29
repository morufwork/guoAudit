"""Cross-protein interaction module.

Models pairwise interaction as a low-rank bilinear form: both embeddings are
projected through the SAME learned matrix U (shared weights, not separate
U/V), then combined by elementwise (Hadamard) product. Sharing U makes the
interaction term symmetric by construction -- U(h_a)*U(h_b) == U(h_b)*U(h_a)
exactly, since elementwise product commutes -- rather than a
generic bilinear form h_a^T W h_b, which would need explicit symmetrization.

The bilinear interaction term is concatenated with the fixed symmetric
statistics (sum, |diff|) so the model has both a learned multiplicative
interaction signal and the simpler additive/difference signal available.
"""
import torch
from torch import nn


class CrossInteractionModule(nn.Module):
    def __init__(self, input_dim: int, interaction_dim: int = 128):
        super().__init__()
        self.project = nn.Linear(input_dim, interaction_dim)  # shared U for both sides
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim + 2 * input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        interaction = self.project(h_a) * self.project(h_b)  # symmetric: shared projection
        summed = h_a + h_b
        diff = torch.abs(h_a - h_b)
        fused = torch.cat([interaction, summed, diff], dim=-1)
        return self.classifier(fused).squeeze(-1)
