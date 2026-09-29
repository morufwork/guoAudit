"""MMseqs2 cluster composition and homology-disjoint splits."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.leakage_checks import run_leakage_checks
from src.utils.io import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [50, 40, 30, 20]

KNOWN_PARALOG_GROUPS = [
    {"P43603", "P43604", "P43605"},
    {"P43612", "P43613"},
    {"Q06407", "Q06408", "Q06409"},
]


def _cluster_file(threshold: int) -> Path:
    return REPO_ROOT / "data" / "processed" / "clusters" / f"cluster_{threshold}.tsv"


@pytest.fixture(scope="module")
def data_config():
    return load_config(REPO_ROOT / "configs" / "data.yaml")


@pytest.fixture(scope="module")
def proteins(data_config):
    path = REPO_ROOT / data_config["proteins_file"]
    return pd.read_csv(path, sep="\t", header=None, names=data_config["proteins_columns"])


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_cluster_files_exist_and_cover_every_protein(threshold, proteins):
    path = _cluster_file(threshold)
    assert path.exists(), f"missing cluster composition for {threshold}%"
    clusters = pd.read_csv(path, sep="\t")
    assert set(clusters["protein_id"]) == set(proteins["protein_id"])


def test_cluster_count_is_nonincreasing_as_identity_threshold_relaxes():
    """A looser (lower) identity threshold can only merge clusters, never split
    them further — the number of clusters must be monotonically non-increasing
    as the threshold decreases from 50% to 20%."""
    counts = []
    for threshold in THRESHOLDS:  # already sorted descending: 50,40,30,20
        clusters = pd.read_csv(_cluster_file(threshold), sep="\t")
        counts.append(clusters["cluster_id"].nunique())
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_hundred_percent_identical_paralogs_stay_clustered_together(threshold):
    """Sequences that are 100% identical must
    always co-cluster, since exact identity exceeds any looser threshold."""
    clusters = pd.read_csv(_cluster_file(threshold), sep="\t")
    id_to_cluster = dict(zip(clusters["protein_id"], clusters["cluster_id"]))
    for group in KNOWN_PARALOG_GROUPS:
        cluster_ids = {id_to_cluster[pid] for pid in group}
        assert len(cluster_ids) == 1, f"{group} split across clusters at {threshold}% identity: {cluster_ids}"


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_saved_homology_split_passes_leakage_checks(threshold):
    split_dir = REPO_ROOT / "data" / "splits" / f"homology_{threshold}"
    train = pd.read_csv(split_dir / "train.tsv", sep="\t")
    test = pd.read_csv(split_dir / "test.tsv", sep="\t")

    clusters = pd.read_csv(_cluster_file(threshold), sep="\t")
    id_to_cluster = dict(zip(clusters["protein_id"], clusters["cluster_id"]))

    run_leakage_checks(f"homology_{threshold}", train, test, id_to_cluster, require_protein_disjoint=True)

    train_proteins = set(train["protein_a"]) | set(train["protein_b"])
    test_proteins = set(test["protein_a"]) | set(test["protein_b"])
    assert train_proteins.isdisjoint(test_proteins)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_homology_split_nonempty_and_balanced(threshold):
    split_dir = REPO_ROOT / "data" / "splits" / f"homology_{threshold}"
    test = pd.read_csv(split_dir / "test.tsv", sep="\t")
    assert len(test) > 0
    balance = test["label"].value_counts(normalize=True)
    assert balance.min() > 0.3, f"severe label imbalance in homology_{threshold} test set: {balance.to_dict()}"


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_no_exact_protein_overlap(threshold):
    split_dir = REPO_ROOT / "data" / "splits" / f"homology_{threshold}"
    train = pd.read_csv(split_dir / "train.tsv", sep="\t")
    test = pd.read_csv(split_dir / "test.tsv", sep="\t")
    train_ids = set(train["protein_a"]) | set(train["protein_b"])
    test_ids = set(test["protein_a"]) | set(test["protein_b"])
    assert train_ids.isdisjoint(test_ids)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_no_mmseqs_cluster_overlap(threshold):
    split_dir = REPO_ROOT / "data" / "splits" / f"homology_{threshold}"
    train = pd.read_csv(split_dir / "train.tsv", sep="\t")
    test = pd.read_csv(split_dir / "test.tsv", sep="\t")
    clusters = pd.read_csv(_cluster_file(threshold), sep="\t").set_index("protein_id")["cluster_id"]
    train_ids = set(train["protein_a"]) | set(train["protein_b"])
    test_ids = set(test["protein_a"]) | set(test["protein_b"])
    assert set(clusters.loc[list(train_ids)]).isdisjoint(set(clusters.loc[list(test_ids)]))


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_pair_reversal_leakage_absent(threshold):
    split_dir = REPO_ROOT / "data" / "splits" / f"homology_{threshold}"
    train = pd.read_csv(split_dir / "train.tsv", sep="\t")
    test = pd.read_csv(split_dir / "test.tsv", sep="\t")
    canon = lambda df: {tuple(sorted(x)) for x in zip(df["protein_a"], df["protein_b"])}
    assert canon(train).isdisjoint(canon(test))


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_no_exact_duplicate_sequence_overlap(threshold, proteins):
    split_dir = REPO_ROOT / "data" / "splits" / f"homology_{threshold}"
    train = pd.read_csv(split_dir / "train.tsv", sep="\t")
    test = pd.read_csv(split_dir / "test.tsv", sep="\t")
    sequence_by_id = proteins.set_index("protein_id")["sequence"]
    train_ids = set(train["protein_a"]) | set(train["protein_b"])
    test_ids = set(test["protein_a"]) | set(test["protein_b"])
    assert set(sequence_by_id.loc[list(train_ids)]).isdisjoint(set(sequence_by_id.loc[list(test_ids)]))
