"""Tests for the negative-bias re-validation: R3 negative substitution must
never reopen the protein-disjointness leakage that the protein-disjoint split closed."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.negative_sampling import _known_pair_set, compute_protein_stats, sample_matched_negatives
from src.data.splits import partition_protein_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42


@pytest.fixture(scope="module")
def proteins():
    return pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")


@pytest.fixture(scope="module")
def pos():
    from src.utils.io import load_config

    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    return pairs[pairs["label"] == 1].reset_index(drop=True)


@pytest.fixture(scope="module")
def train_test_pool(proteins):
    return partition_protein_pool(proteins, 0.20, SEED)


def test_matched_negatives_for_r3_stay_protein_disjoint(proteins, pos, train_test_pool):
    train_pool, test_pool = train_test_pool
    stats = compute_protein_stats(proteins, pos, n_deciles=10)
    known = _known_pair_set(pos)

    r3_train_raw = pd.read_csv(REPO_ROOT / "data" / "splits" / "both_unseen" / "train.tsv", sep="\t")
    r3_test_raw = pd.read_csv(REPO_ROOT / "data" / "splits" / "both_unseen" / "test.tsv", sep="\t")
    r3_train_pos = r3_train_raw[r3_train_raw["label"] == 1]
    r3_test_pos = r3_test_raw[r3_test_raw["label"] == 1]

    for cols in (["degree_decile"], ["degree_decile", "length_decile"]):
        train_neg = sample_matched_negatives(r3_train_pos, stats, known, cols, SEED, candidate_pool=train_pool)
        test_neg = sample_matched_negatives(r3_test_pos, stats, known, cols, SEED, candidate_pool=test_pool)

        train_endpoints = set(train_neg["protein_a"]) | set(train_neg["protein_b"])
        test_endpoints = set(test_neg["protein_a"]) | set(test_neg["protein_b"])
        assert train_endpoints.issubset(train_pool)
        assert test_endpoints.issubset(test_pool)
        assert train_endpoints.isdisjoint(test_endpoints)


def test_revalidation_results_reproduce_representation_comparison_r3_n0_numbers():
    """R3/N0 in the revalidation table must exactly match the representation comparison's own
    numbers (same data, same code path) -- a cross-script consistency guard."""
    path = REPO_ROOT / "tables" / "negative_bias_revalidation_summary.csv"
    if not path.exists():
        pytest.skip("negative_bias_revalidation_summary.csv not present")
    revalidation = pd.read_csv(path)
    row = revalidation[(revalidation["split"] == "both_unseen") & (revalidation["negative_set"] == "n0") & (revalidation["representation"] == "esm2_mean")]
    assert row["roc_auc"].iloc[0] == pytest.approx(0.6168, abs=0.001)
