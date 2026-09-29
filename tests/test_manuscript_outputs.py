"""Tests for manuscript output consolidation (Table 1, Table 9 status).

Includes a regression guard for a real error caught 2026-08-23: the sequence-cluster split's
original write-up claimed ~7.7% of proteins have any homolog at 20%
identity, computed (incorrectly) from cluster-count reduction
(1 - n_clusters/n_proteins). The correct, cluster-membership-based figure
is 20.2% under the explicitly versioned regenerated clustering -- this pins that number down directly from the saved cluster
assignment file so the error cannot silently return.
"""
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fraction_with_homolog_matches_verified_cluster_membership_count():
    cluster_path = REPO_ROOT / "data" / "processed" / "clusters" / "cluster_20.tsv"
    if not cluster_path.exists():
        pytest.skip("cluster_20.tsv not present")

    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from importlib import import_module

    mod = import_module("21_generate_manuscript_outputs")
    frac = mod._fraction_with_any_homolog(cluster_path)

    # Verified independently after the versioned, deterministic the dataset audit
    # regeneration (2026-08-28).
    assert frac == pytest.approx(0.2022, abs=0.001)
    # And explicitly NOT the erroneous cluster-count-reduction proxy that
    # produced ~7.7% in the original (incorrect) the sequence-cluster split write-up.
    assert abs(frac - 0.077) > 0.05


def test_cluster_count_reduction_is_not_a_valid_proxy_for_homolog_fraction():
    """Documents *why* the original error happened: cluster-count reduction
    (1 - n_clusters/n_proteins) is a different, larger-magnitude quantity
    than the fraction of proteins actually sharing a cluster, and the two
    must not be conflated again."""
    cluster_path = REPO_ROOT / "data" / "processed" / "clusters" / "cluster_20.tsv"
    if not cluster_path.exists():
        pytest.skip("cluster_20.tsv not present")

    df = pd.read_csv(cluster_path, sep="\t")
    n_proteins = len(df)
    n_clusters = df["cluster_id"].nunique()
    naive_proxy = 1 - (n_clusters / n_proteins)

    cluster_sizes = df.groupby("cluster_id").size()
    correct_fraction = cluster_sizes[cluster_sizes > 1].sum() / n_proteins

    assert naive_proxy != pytest.approx(correct_fraction, abs=0.01)


def test_table1_exists_and_has_expected_dataset_dimensions():
    path = REPO_ROOT / "tables" / "table1_dataset_characteristics.csv"
    if not path.exists():
        pytest.skip("table1_dataset_characteristics.csv not present -- run scripts/21_generate_manuscript_outputs.py")
    df = pd.read_csv(path).set_index("Characteristic")["Value"]
    assert df["Unique proteins (raw)"] == 2497
    assert df["PPI pairs (raw)"] == 11188
    assert df["Canonical (deduplicated) pairs"] == 11164


def test_table9_status_documents_deliberate_skip():
    path = REPO_ROOT / "tables" / "table9_external_validation_status.json"
    if not path.exists():
        pytest.skip("table9_external_validation_status.json not present")
    import json
    status = json.load(open(path))
    assert status["status"] == "not_applicable_by_design"
    assert "reason" in status and len(status["reason"]) > 0
