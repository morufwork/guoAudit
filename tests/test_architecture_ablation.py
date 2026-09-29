"""Attention fusion and cross-interaction module."""
import numpy as np
import pytest

from src.models.attention import AttentionFusion
from src.models.interaction import CrossInteractionModule
from src.models.torch_utils import predict_proba, train_pair_model

ARCHITECTURES = {
    "attention": lambda dim: AttentionFusion(dim, proj_dim=32),
    "interaction": lambda dim: CrossInteractionModule(dim, interaction_dim=32),
}


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_architecture_can_overfit_a_tiny_toy_dataset(name):
    rng = np.random.default_rng(0)
    n, dim = 60, 16
    h_a = rng.normal(size=(n, dim)).astype(np.float32)
    h_b = rng.normal(size=(n, dim)).astype(np.float32)
    y = ((h_a.mean(axis=1) + h_b.mean(axis=1)) > 0).astype(np.float32)

    model_fn = ARCHITECTURES[name]
    model = train_pair_model(lambda: model_fn(dim), h_a, h_b, y, seed=0, max_epochs=200, patience=50, val_fraction=0.2)
    y_prob = predict_proba(model, h_a, h_b)
    accuracy = ((y_prob >= 0.5).astype(int) == y).mean()
    assert accuracy > 0.85, f"{name} failed to overfit trivially separable toy data: accuracy={accuracy}"


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_architecture_is_exactly_pair_order_invariant(name):
    rng = np.random.default_rng(1)
    n, dim = 30, 16
    h_a = rng.normal(size=(n, dim)).astype(np.float32)
    h_b = rng.normal(size=(n, dim)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.float32)

    model_fn = ARCHITECTURES[name]
    model = train_pair_model(lambda: model_fn(dim), h_a, h_b, y, seed=1, max_epochs=10)
    p_forward = predict_proba(model, h_a, h_b)
    p_swapped = predict_proba(model, h_b, h_a)
    np.testing.assert_array_equal(p_forward, p_swapped)


def test_train_pair_model_seeds_before_construction():
    """train_pair_model must build the model itself (via model_fn) so seeding
    happens before weight initialization -- passing an already-built model
    would silently depend on ambient RNG state (an earlier bug)."""
    import inspect

    sig = inspect.signature(train_pair_model)
    first_param = next(iter(sig.parameters))
    assert first_param == "model_fn"
