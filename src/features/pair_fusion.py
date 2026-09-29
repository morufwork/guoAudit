"""Symmetric pair-fusion operations.

Given protein embeddings h_A, h_B, these combine them so the result is
invariant to input order — required so P(A,B) == P(B,A) for an undirected
PPI. Concatenation [A,B] is deliberately NOT provided here since
it is order-sensitive; see scripts/10_pair_fusion_ablation.py for the full ablation.
"""
import numpy as np


def fusion_sum(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    return h_a + h_b


def fusion_abs_diff(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    return np.abs(h_a - h_b)


def fusion_hadamard(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    return h_a * h_b


def fusion_diff_product(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    return np.concatenate([fusion_abs_diff(h_a, h_b), fusion_hadamard(h_a, h_b)], axis=-1)


def fusion_combined(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    """Recommended default: [sum, |diff|, product]."""
    return np.concatenate([fusion_sum(h_a, h_b), fusion_abs_diff(h_a, h_b), fusion_hadamard(h_a, h_b)], axis=-1)


def fusion_concat(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    """NOT symmetric: fusion_concat(A,B) != fusion_concat(B,A). Included only
    for the pair-fusion ablation, which needs it as the explicit counter-example
    of an order-sensitive operator — never use for a model
    claimed to be pair-order invariant."""
    return np.concatenate([h_a, h_b], axis=-1)


SYMMETRIC_FUSIONS = {
    "sum": fusion_sum,
    "abs_diff": fusion_abs_diff,
    "hadamard": fusion_hadamard,
    "combined": fusion_combined,
}

# Superset used only by the pair-fusion ablation script — includes the
# order-sensitive fusion_concat alongside every symmetric option, plus the
# diff+product combination. Never iterate this
# dict in an order-invariance test (see SYMMETRIC_FUSIONS for that).
ABLATION_FUSIONS = {
    **SYMMETRIC_FUSIONS,
    "diff_product": fusion_diff_product,
    "concat": fusion_concat,
}
