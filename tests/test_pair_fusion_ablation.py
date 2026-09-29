"""Ablation fusion set and learned symmetric fusion."""
import numpy as np
import pytest
import torch

from src.features.pair_fusion import ABLATION_FUSIONS, SYMMETRIC_FUSIONS, fusion_concat
from src.models.siamese import predict_proba, train_shared_encoder_fusion


def test_ablation_fusions_is_superset_of_symmetric_fusions():
    assert set(SYMMETRIC_FUSIONS).issubset(set(ABLATION_FUSIONS))
    assert "concat" in ABLATION_FUSIONS
    assert "diff_product" in ABLATION_FUSIONS


def test_concat_is_deliberately_not_order_invariant():
    """Sanity check that fusion_concat correctly represents the 'bad example'
    of an order-sensitive operator -- if this ever became
    order-invariant, the whole point of including it in the ablation breaks."""
    rng = np.random.default_rng(0)
    h_a = rng.normal(size=8)
    h_b = rng.normal(size=8)
    assert not np.allclose(fusion_concat(h_a, h_b), fusion_concat(h_b, h_a))


def test_shared_encoder_fusion_can_overfit_a_tiny_toy_dataset():
    """Plan Section 14 'Model tests': model can overfit a tiny toy dataset."""
    rng = np.random.default_rng(0)
    n, dim = 40, 8
    h_a = rng.normal(size=(n, dim)).astype(np.float32)
    h_b = rng.normal(size=(n, dim)).astype(np.float32)
    # trivially separable: label = 1 iff sum of means is positive
    y = ((h_a.mean(axis=1) + h_b.mean(axis=1)) > 0).astype(np.float32)

    model = train_shared_encoder_fusion(h_a, h_b, y, input_dim=dim, seed=0, max_epochs=200, patience=50, val_fraction=0.2)
    y_prob = predict_proba(model, h_a, h_b)
    accuracy = ((y_prob >= 0.5).astype(int) == y).mean()
    assert accuracy > 0.85, f"failed to overfit trivially separable toy data: accuracy={accuracy}"


def test_shared_encoder_fusion_training_is_independent_of_ambient_rng_state():
    """Regression guard: train_pair_model must seed BEFORE constructing the
    model, or weight initialization silently depends on whatever unrelated
    torch calls happened earlier in the process (a real bug caught while
    building the architecture ablation, where a refactor moved model construction outside
    the seeded region)."""
    rng = np.random.default_rng(0)
    h_a = rng.normal(size=(40, 8)).astype(np.float32)
    h_b = rng.normal(size=(40, 8)).astype(np.float32)
    y = rng.integers(0, 2, size=40).astype(np.float32)

    torch.manual_seed(999)
    _ = torch.randn(1000, 1000)  # pollute ambient RNG state
    m1 = train_shared_encoder_fusion(h_a, h_b, y, input_dim=8, seed=42, max_epochs=20)
    p1 = predict_proba(m1, h_a, h_b)

    torch.manual_seed(12345)
    _ = torch.randn(500, 500)
    _ = torch.randn(77)  # pollute differently
    m2 = train_shared_encoder_fusion(h_a, h_b, y, input_dim=8, seed=42, max_epochs=20)
    p2 = predict_proba(m2, h_a, h_b)

    np.testing.assert_array_equal(p1, p2)


def test_shared_encoder_fusion_is_exactly_pair_order_invariant():
    rng = np.random.default_rng(1)
    n, dim = 30, 8
    h_a = rng.normal(size=(n, dim)).astype(np.float32)
    h_b = rng.normal(size=(n, dim)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.float32)

    model = train_shared_encoder_fusion(h_a, h_b, y, input_dim=dim, seed=1, max_epochs=10)
    p_forward = predict_proba(model, h_a, h_b)
    p_swapped = predict_proba(model, h_b, h_a)
    np.testing.assert_array_equal(p_forward, p_swapped)
