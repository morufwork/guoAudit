"""Learned symmetric fusion ("Siamese MLP" in the architecture ablation).

A shared-weight encoder phi() is applied independently to each protein
embedding; the two outputs are combined by a commutative sum, so the model
is symmetric by construction (P(A,B) == P(B,A) exactly, not just
approximately) rather than symmetric because of a fixed pre-combination
statistic (sum/diff/product) the way the other fixed-fusion candidates are.
This is a genuinely different structural hypothesis from "combined+MLP" --
the latter already lets its first layer learn how to weight fixed pairwise
statistics, whereas this learns a nonlinear transform of each embedding
BEFORE any combination happens.
"""
import numpy as np
import torch
from torch import nn

from src.models.torch_utils import predict_proba as _predict_proba
from src.models.torch_utils import train_pair_model


class SharedEncoderFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        fused = self.phi(h_a) + self.phi(h_b)  # commutative -> exactly order-invariant
        return self.classifier(fused).squeeze(-1)


def train_shared_encoder_fusion(
    h_a_train: np.ndarray, h_b_train: np.ndarray, y_train: np.ndarray,
    input_dim: int, seed: int, hidden_dim: int = 256, latent_dim: int = 128, **kwargs,
) -> SharedEncoderFusion:
    """Thin wrapper over the generic trainer, kept for backward compatibility
    with existing callers/tests."""
    return train_pair_model(
        lambda: SharedEncoderFusion(input_dim, hidden_dim, latent_dim),
        h_a_train, h_b_train, y_train, seed, **kwargs,
    )


def predict_proba(model: SharedEncoderFusion, h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    return _predict_proba(model, h_a, h_b)
