"""Dataset QC tests: required before any splitting/training."""
from pathlib import Path

import pytest

from src.data import validate_data
from src.data.load_data import load_original_split, load_pairs, load_proteins
from src.utils.io import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_config(REPO_ROOT / "configs" / "data.yaml")


@pytest.fixture(scope="module")
def proteins(config):
    return load_proteins(config, REPO_ROOT)


@pytest.fixture(scope="module")
def pairs(config):
    return load_pairs(config, REPO_ROOT)


def test_expected_dataset_dimensions(config, proteins, pairs):
    assert proteins["protein_id"].nunique() == config["expected"]["n_proteins"]
    assert len(pairs) == config["expected"]["n_pairs"]
    assert (pairs["label"] == 1).sum() == config["expected"]["n_positive"]
    assert (pairs["label"] == 0).sum() == config["expected"]["n_negative"]


def test_no_missing_sequences_or_ids(proteins):
    assert len(validate_data.missing_sequences(proteins)) == 0
    assert len(validate_data.missing_ids(proteins)) == 0


def test_no_invalid_amino_acids(proteins, config):
    assert len(validate_data.invalid_amino_acids(proteins, config["valid_amino_acids"])) == 0


def test_no_duplicate_protein_ids(proteins):
    assert len(validate_data.duplicate_protein_ids(proteins)) == 0


def test_no_conflicting_duplicate_pairs(pairs):
    """Duplicate/reverse pairs may exist (see reverse_duplicate_pairs), but
    none may carry conflicting labels — that would indicate corrupted data."""
    assert len(validate_data.conflicting_labels(pairs)) == 0


def test_no_self_interactions(pairs):
    assert len(validate_data.self_interactions(pairs)) == 0


def test_labels_are_binary(pairs):
    assert set(pairs["label"].unique()) == {0, 1}


def test_all_paired_proteins_exist_in_dictionary(pairs, proteins):
    assert len(validate_data.pairs_referencing_unknown_proteins(pairs, proteins)) == 0


def test_original_split_reconstructs_full_pairs_file(config, pairs):
    train, test = load_original_split(config, REPO_ROOT)
    combined = len(train) + len(test)
    assert combined == len(pairs)


def test_sequences_map_consistently_to_protein_ids(proteins):
    """A given protein_id must always map to exactly one sequence."""
    grouped = proteins.groupby("protein_id")["sequence"].nunique()
    assert (grouped == 1).all()
