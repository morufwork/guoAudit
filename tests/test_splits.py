"""Split generation and leakage checks."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.leakage_checks import (
    LeakageError,
    check_group_disjointness,
    check_label_integrity,
    check_pair_duplicate_leakage,
    check_protein_disjointness,
    check_reverse_pair_leakage,
    run_leakage_checks,
)
from src.data.splits import make_r0_random_split, make_r1_r2_r3_splits, partition_protein_pool
from src.utils.io import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42


@pytest.fixture(scope="module")
def data_config():
    return load_config(REPO_ROOT / "configs" / "data.yaml")


@pytest.fixture(scope="module")
def pairs(data_config):
    return pd.read_csv(
        REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"]
    )


@pytest.fixture(scope="module")
def proteins_with_groups():
    return pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")


@pytest.fixture(scope="module")
def id_to_group(proteins_with_groups):
    return dict(zip(proteins_with_groups["protein_id"], proteins_with_groups["sequence_group_id"]))


@pytest.fixture(scope="module")
def r1_r2_r3(pairs, proteins_with_groups):
    return make_r1_r2_r3_splits(pairs, proteins_with_groups, test_protein_pool_fraction=0.2, r1_holdout_fraction=0.2, seed=SEED)


def test_protein_pool_partition_respects_sequence_groups(proteins_with_groups):
    train_pool, test_pool = partition_protein_pool(proteins_with_groups, 0.2, SEED)
    id_to_group_local = dict(zip(proteins_with_groups["protein_id"], proteins_with_groups["sequence_group_id"]))
    train_groups = {id_to_group_local[p] for p in train_pool}
    test_groups = {id_to_group_local[p] for p in test_pool}
    assert train_groups.isdisjoint(test_groups)
    # the 3 known paralog groups must not be split
    for group in [{"P43603", "P43604", "P43605"}, {"P43612", "P43613"}, {"Q06407", "Q06408", "Q06409"}]:
        assert group.issubset(train_pool) or group.issubset(test_pool)


def test_r0_random_split_has_no_pair_overlap(pairs):
    train, test = make_r0_random_split(pairs, 0.2, SEED)
    assert check_pair_duplicate_leakage(train, test)
    assert check_reverse_pair_leakage(train, test)
    assert check_label_integrity(train, test)
    assert len(train) + len(test) == len(pairs)


def test_r0_permits_protein_overlap_by_design(pairs):
    """R0 is the naive baseline — protein overlap is expected, not a bug."""
    train, test = make_r0_random_split(pairs, 0.2, SEED)
    train_p = set(train["protein_a"]) | set(train["protein_b"])
    test_p = set(test["protein_a"]) | set(test["protein_b"])
    assert len(train_p & test_p) > 0


def test_seen_seen_train_and_test_are_pairwise_disjoint(r1_r2_r3):
    train, test = r1_r2_r3["seen_seen"]
    assert check_pair_duplicate_leakage(train, test)
    assert check_reverse_pair_leakage(train, test)


def test_seen_seen_test_proteins_all_individually_seen_in_training(r1_r2_r3):
    """R1: both proteins of every test pair must appear somewhere in the training set."""
    train, test = r1_r2_r3["seen_seen"]
    train_proteins = set(train["protein_a"]) | set(train["protein_b"])
    for _, row in test.iterrows():
        assert row["protein_a"] in train_proteins
        assert row["protein_b"] in train_proteins


def test_one_unseen_exactly_one_protein_absent_from_training(r1_r2_r3):
    """R2: exactly one member of every test pair is absent from training."""
    train, test = r1_r2_r3["one_unseen"]
    train_proteins = set(train["protein_a"]) | set(train["protein_b"])
    for _, row in test.iterrows():
        a_seen = row["protein_a"] in train_proteins
        b_seen = row["protein_b"] in train_proteins
        assert a_seen != b_seen, f"pair ({row['protein_a']},{row['protein_b']}) should have exactly one seen protein"


def test_both_unseen_is_fully_protein_disjoint(r1_r2_r3, id_to_group):
    """R3: neither test protein occurs in training. The key de novo evaluation."""
    train, test = r1_r2_r3["both_unseen"]
    assert check_protein_disjointness(train, test, required=True)
    assert check_group_disjointness(train, test, id_to_group, required=True)


def test_both_unseen_test_set_nonempty(r1_r2_r3):
    _, test = r1_r2_r3["both_unseen"]
    assert len(test) > 0


def test_seen_seen_one_unseen_both_unseen_share_identical_training_set(r1_r2_r3):
    """Comparing R1/R2/R3 requires identical training data."""
    train_r1, _ = r1_r2_r3["seen_seen"]
    train_r2, _ = r1_r2_r3["one_unseen"]
    train_r3, _ = r1_r2_r3["both_unseen"]
    pd.testing.assert_frame_equal(train_r1, train_r2)
    pd.testing.assert_frame_equal(train_r1, train_r3)


def test_run_leakage_checks_raises_on_injected_pair_duplicate(pairs, id_to_group):
    train, test = make_r0_random_split(pairs, 0.2, SEED)
    contaminated_test = pd.concat([test, train.iloc[[0]]], ignore_index=True)
    with pytest.raises(LeakageError):
        run_leakage_checks("contaminated", train, contaminated_test, id_to_group, require_protein_disjoint=False)


def test_run_leakage_checks_raises_on_injected_protein_overlap_when_required(r1_r2_r3, id_to_group):
    train, test = r1_r2_r3["both_unseen"]
    # inject a test row that reuses a training protein — should be caught even though
    # pair-level duplicate/reverse checks alone would not catch it.
    contaminated_test = pd.concat(
        [test, pd.DataFrame({"protein_a": [train.iloc[0]["protein_a"]], "protein_b": [test.iloc[0]["protein_b"]], "label": [0]})],
        ignore_index=True,
    )
    with pytest.raises(LeakageError):
        run_leakage_checks("contaminated_r3", train, contaminated_test, id_to_group, require_protein_disjoint=True)


def test_saved_split_files_match_generation_and_pass_checks(data_config, id_to_group):
    """Regression guard: files under data/splits/ must exist and still pass checks
    (i.e. were not hand-edited or regenerated inconsistently)."""
    splits_dir = REPO_ROOT / "data" / "splits"
    specs = {"random": False, "seen_seen": False, "one_unseen": False, "both_unseen": True}
    for name, require_disjoint in specs.items():
        train_path = splits_dir / name / "train.tsv"
        test_path = splits_dir / name / "test.tsv"
        assert train_path.exists() and test_path.exists(), f"missing saved split files for {name}"
        train = pd.read_csv(train_path, sep="\t")
        test = pd.read_csv(test_path, sep="\t")
        run_leakage_checks(name, train, test, id_to_group, require_disjoint)
