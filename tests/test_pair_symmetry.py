"""Canonical pair ordering, dedup, symmetric fusion, sequence groups."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.canonicalize import canonicalize_pairs
from src.data.load_data import load_pairs, load_proteins
from src.data.sequence_groups import assign_sequence_groups
from src.features.pair_fusion import SYMMETRIC_FUSIONS
from src.utils.io import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_config(REPO_ROOT / "configs" / "data.yaml")


@pytest.fixture(scope="module")
def pairs(config):
    return load_pairs(config, REPO_ROOT)


@pytest.fixture(scope="module")
def proteins(config):
    return load_proteins(config, REPO_ROOT)


def test_canonical_ordering_is_alphabetical(pairs):
    clean = canonicalize_pairs(pairs)
    assert (clean["protein_a"] < clean["protein_b"]).all()


def test_canonicalization_drops_exactly_the_known_duplicates(pairs):
    clean = canonicalize_pairs(pairs)
    assert clean.attrs["n_dropped_duplicates"] == 24
    assert len(clean) == len(pairs) - 24


def test_canonicalization_preserves_label_balance_minus_duplicates(pairs):
    clean = canonicalize_pairs(pairs)
    # the 24 dropped rows had no label conflicts (verified in the dataset audit),
    # so canonicalization must not change the *set* of (pair, label) facts.
    orig_canon = pairs.apply(
        lambda r: (r["protein_a"], r["protein_b"], r["label"]) if r["protein_a"] < r["protein_b"]
        else (r["protein_b"], r["protein_a"], r["label"]),
        axis=1,
    )
    assert set(orig_canon) == set(clean.apply(lambda r: (r["protein_a"], r["protein_b"], r["label"]), axis=1))


def test_canonicalization_is_idempotent(pairs):
    once = canonicalize_pairs(pairs)
    twice = canonicalize_pairs(once)
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True))


@pytest.mark.parametrize("name", ["sum", "abs_diff", "hadamard", "combined"])
def test_symmetric_fusion_is_order_invariant(name):
    rng = np.random.default_rng(0)
    h_a = rng.normal(size=16)
    h_b = rng.normal(size=16)
    fn = SYMMETRIC_FUSIONS[name]
    np.testing.assert_allclose(fn(h_a, h_b), fn(h_b, h_a), atol=1e-12)


def test_sequence_groups_unite_known_identical_sequence_proteins(proteins):
    grouped = assign_sequence_groups(proteins)
    known_groups = [
        {"P43603", "P43604", "P43605"},
        {"P43612", "P43613"},
        {"Q06407", "Q06408", "Q06409"},
    ]
    id_to_group = dict(zip(grouped["protein_id"], grouped["sequence_group_id"]))
    for group in known_groups:
        group_ids = {id_to_group[pid] for pid in group}
        assert len(group_ids) == 1, f"{group} should share one sequence_group_id, got {group_ids}"


def test_sequence_groups_cover_every_protein_exactly_once(proteins):
    grouped = assign_sequence_groups(proteins)
    assert len(grouped) == len(proteins)
    assert grouped["protein_id"].nunique() == len(proteins)


def test_distinct_sequences_get_distinct_groups(proteins):
    grouped = assign_sequence_groups(proteins)
    n_unique_sequences = proteins["sequence"].nunique()
    assert grouped["sequence_group_id"].nunique() == n_unique_sequences
