"""Tests for the ProtT5 cross-check: (1) the dataset-builder functions moved
out of scripts/14_negative_bias_revalidation.py into
src/data/negative_sampling.py still produce the exact same dataset sizes
after the refactor, and (2) once run, the cross-check's re-derived
AAC+CTD/ESM-2 numbers exactly match scripts/14's already-verified table --
proving this second harness didn't silently diverge from the first."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.negative_sampling import (
    _known_pair_set,
    build_r0_n0_n2_n5_datasets,
    build_r3_n0_n2_n5_datasets,
    compute_protein_stats,
)
from src.utils.io import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42


@pytest.fixture(scope="module")
def proteins():
    return pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")


@pytest.fixture(scope="module")
def pairs():
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    return pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])


def test_r0_dataset_builder_sizes_match_known_revalidation_output(proteins, pairs):
    """These sizes are exactly what scripts/14's original (pre-refactor)
    log output reported for R0/N0 -- a regression guard on the
    negative_sampling.py extraction, independent of re-running the full
    script."""
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)
    stats = compute_protein_stats(proteins, pos, n_deciles=10)
    known = _known_pair_set(pos)

    datasets = build_r0_n0_n2_n5_datasets(pos, neg0, stats, known, SEED, 0.20)
    train_n0, test_n0 = datasets["n0"]
    assert len(train_n0) == 8930
    assert len(test_n0) == 2234


def test_r3_dataset_builder_sizes_match_known_revalidation_output(proteins, pairs):
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    stats = compute_protein_stats(proteins, pos, n_deciles=10)
    known = _known_pair_set(pos)

    datasets = build_r3_n0_n2_n5_datasets(stats, known, proteins, REPO_ROOT, SEED, 0.20)
    train_n0, test_n0 = datasets["n0"]
    assert len(train_n0) == 5703
    assert len(test_n0) == 462


def test_cross_check_reproduces_negative_bias_revalidation_for_shared_representations():
    cross_check_path = REPO_ROOT / "tables" / "prott5_cross_check_summary.csv"
    revalidation_path = REPO_ROOT / "tables" / "negative_bias_revalidation_summary.csv"
    if not cross_check_path.exists() or not revalidation_path.exists():
        pytest.skip("prott5_cross_check_summary.csv not present -- run scripts/20_prott5_cross_check.py")

    cross_check = pd.read_csv(cross_check_path)
    revalidation = pd.read_csv(revalidation_path)

    for repr_name in ["aac_ctd", "esm2_mean"]:
        for split in ["random", "both_unseen"]:
            for neg in ["n0", "n2", "n5"]:
                cc_row = cross_check[(cross_check.representation == repr_name) & (cross_check.split == split) & (cross_check.negative_set == neg)]
                rv_row = revalidation[(revalidation.representation == repr_name) & (revalidation.split == split) & (revalidation.negative_set == neg)]
                assert cc_row["roc_auc"].iloc[0] == pytest.approx(rv_row["roc_auc"].iloc[0], abs=1e-9), (
                    f"{repr_name}/{split}/{neg}: cross-check harness diverged from scripts/14's verified numbers"
                )


def test_prott5_mean_present_with_valid_metrics_in_cross_check():
    path = REPO_ROOT / "tables" / "prott5_cross_check_summary.csv"
    if not path.exists():
        pytest.skip("prott5_cross_check_summary.csv not present -- run scripts/20_prott5_cross_check.py")
    df = pd.read_csv(path)
    prott5_rows = df[df.representation == "prott5_mean"]
    assert len(prott5_rows) == 6  # 2 splits x 3 negative sets
    assert prott5_rows["roc_auc"].between(0, 1).all()
    assert not prott5_rows["roc_auc"].isna().any()
