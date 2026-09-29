"""Negative-set construction correctness."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.negative_sampling import (
    _known_pair_set,
    compute_protein_stats,
    sample_matched_negatives,
    sample_nearest_degree_matched_negatives,
    sample_random_negatives,
    sample_two_sided_matched_negatives,
)
from src.utils.io import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42


@pytest.fixture(scope="module")
def data_config():
    return load_config(REPO_ROOT / "configs" / "data.yaml")


@pytest.fixture(scope="module")
def pos(data_config):
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    return pairs[pairs["label"] == 1].reset_index(drop=True)


@pytest.fixture(scope="module")
def proteins():
    return pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")


@pytest.fixture(scope="module")
def protein_stats(proteins, pos):
    return compute_protein_stats(proteins, pos, n_deciles=10)


@pytest.fixture(scope="module")
def known_pairs(pos):
    return _known_pair_set(pos)


def _pair_stat_mean(df, stats, col):
    return pd.concat([df["protein_a"].map(stats[col]), df["protein_b"].map(stats[col])]).mean()


def test_no_generated_negative_coincides_with_a_known_positive(pos, protein_stats, known_pairs):
    n1 = sample_random_negatives(list(protein_stats.index), known_pairs, len(pos), SEED)
    n2 = sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], SEED)
    n3 = sample_matched_negatives(pos, protein_stats, known_pairs, ["length_decile"], SEED)
    n5 = sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile", "length_decile"], SEED)

    for name, df in [("n1", n1), ("n2", n2), ("n3", n3), ("n5", n5)]:
        overlap = set(zip(df["protein_a"], df["protein_b"])) & known_pairs
        assert not overlap, f"{name} contains {len(overlap)} pairs identical to known positives"


def test_no_self_pairs_in_any_negative_set(pos, protein_stats, known_pairs):
    n1 = sample_random_negatives(list(protein_stats.index), known_pairs, len(pos), SEED)
    n2 = sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], SEED)
    for df in (n1, n2):
        assert (df["protein_a"] != df["protein_b"]).all()


def test_degree_matched_negatives_closely_match_positive_degree_distribution(pos, protein_stats, known_pairs):
    """N2 must remove the degree confound: mean positive-degree of its
    endpoints should land within ~5% of the positive set's mean."""
    n2 = sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], SEED)
    pos_mean = _pair_stat_mean(pos, protein_stats, "positive_degree")
    n2_mean = _pair_stat_mean(n2, protein_stats, "positive_degree")
    assert abs(n2_mean - pos_mean) / pos_mean < 0.05


def test_length_matched_negatives_closely_match_positive_length_distribution(pos, protein_stats, known_pairs):
    n3 = sample_matched_negatives(pos, protein_stats, known_pairs, ["length_decile"], SEED)
    pos_mean = _pair_stat_mean(pos, protein_stats, "length")
    n3_mean = _pair_stat_mean(n3, protein_stats, "length")
    assert abs(n3_mean - pos_mean) / pos_mean < 0.05


def test_hard_negatives_match_both_degree_and_length(pos, protein_stats, known_pairs):
    n5 = sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile", "length_decile"], SEED)
    pos_degree_mean = _pair_stat_mean(pos, protein_stats, "positive_degree")
    pos_length_mean = _pair_stat_mean(pos, protein_stats, "length")
    n5_degree_mean = _pair_stat_mean(n5, protein_stats, "positive_degree")
    n5_length_mean = _pair_stat_mean(n5, protein_stats, "length")
    assert abs(n5_degree_mean - pos_degree_mean) / pos_degree_mean < 0.05
    assert abs(n5_length_mean - pos_length_mean) / pos_length_mean < 0.05


def test_original_benchmark_negatives_do_not_match_positive_degree_distribution(pos, protein_stats, data_config):
    """Documents the actual confound driving the negative-sampling study's central finding: N0's
    own negatives are NOT degree-matched to positives (mean degree ~half)."""
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    neg0 = pairs[pairs["label"] == 0]
    pos_mean = _pair_stat_mean(pos, protein_stats, "positive_degree")
    neg0_mean = _pair_stat_mean(neg0, protein_stats, "positive_degree")
    assert neg0_mean < pos_mean * 0.7, (
        f"expected N0's negatives to show a substantial degree confound relative to positives "
        f"(pos={pos_mean:.2f}, neg0={neg0_mean:.2f}) -- if this changes, the negative-sampling study's central "
        f"finding needs re-examination, not just this test updated"
    )


def test_matched_negative_sizes_are_at_most_the_positive_count(pos, protein_stats, known_pairs):
    for cols in (["degree_decile"], ["length_decile"], ["degree_decile", "length_decile"]):
        n = sample_matched_negatives(pos, protein_stats, known_pairs, cols, SEED)
        assert 0 < len(n) <= len(pos)


def test_nearest_degree_matched_negatives_do_not_overconcentrate_on_one_decoy(pos, protein_stats, known_pairs):
    """Regression test for a matching-algorithm artifact found on the
    multi-species replication: an
    earlier version of sample_nearest_degree_matched_negatives walked a
    single globally-sorted array with a deterministic lo/hi tie-break, so
    every query sharing a target degree tried the same candidate first --
    one decoy protein ended up reused in 1,378/31,678 (~4.3%) of that
    dataset's generated negatives, letting sequence-based models learn
    decoy identity instead of anything about degree-matching quality.
    Per-query random tie-breaking should keep any single decoy's reuse well
    under that on this benchmark too."""
    n2_nn = sample_nearest_degree_matched_negatives(pos, protein_stats, known_pairs, SEED)
    assert len(n2_nn) > 0
    assert n2_nn.attrs["max_decoy_reuse_fraction"] < 0.02, (
        f"a single decoy protein accounts for {n2_nn.attrs['max_decoy_reuse_fraction']:.1%} of N2-nn's "
        f"generated negatives (count={n2_nn.attrs['max_decoy_reuse_count']}/{len(n2_nn)}) -- this is the "
        f"decoy-overconcentration artifact sample_nearest_degree_matched_negatives was fixed to avoid"
    )


# --- N6: two-sided negative construction ---

def test_two_sided_negatives_no_overlap_with_known_positives_or_self_pairs(pos, protein_stats, known_pairs):
    n6 = sample_two_sided_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], SEED)
    overlap = set(zip(n6["protein_a"], n6["protein_b"])) & known_pairs
    assert not overlap
    assert (n6["protein_a"] != n6["protein_b"]).all()


def test_two_sided_negatives_match_positive_degree_distribution(pos, protein_stats, known_pairs):
    n6 = sample_two_sided_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], SEED)
    pos_mean = _pair_stat_mean(pos, protein_stats, "positive_degree")
    n6_mean = _pair_stat_mean(n6, protein_stats, "positive_degree")
    assert abs(n6_mean - pos_mean) / pos_mean < 0.05


def test_two_sided_negatives_size_bounded_by_positive_count(pos, protein_stats, known_pairs):
    n6 = sample_two_sided_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], SEED)
    assert 0 < len(n6) <= len(pos)


def test_two_sided_negatives_need_not_anchor_either_endpoint_at_the_source_positive():
    """The defining structural difference from `sample_matched_negatives`
    (N2): by construction, every N2 decoy keeps one endpoint fixed at a real
    protein from its own source positive pair (u,v) and substitutes only
    the other -- so a decoy for (u,v) always contains u or v. N6 draws BOTH
    decoy endpoints fresh, so a decoy for (u,v) need not contain u or v at
    all. Uses a small synthetic pool (not the real dataset) so the outcome
    is checked directly rather than statistically."""
    proteins = [f"P{i}" for i in range(1, 21)]
    stats = pd.DataFrame({"protein_id": proteins, "degree_decile": 0}).set_index("protein_id")  # one shared bucket
    positive_pairs = pd.DataFrame({"protein_a": ["P1"], "protein_b": ["P2"]})
    known_pairs = {("P1", "P2"), ("P2", "P1")}

    n2_decoys = set()
    n6_decoys = set()
    for seed in range(30):
        n2 = sample_matched_negatives(positive_pairs, stats, known_pairs, ["degree_decile"], seed)
        n6 = sample_two_sided_matched_negatives(positive_pairs, stats, known_pairs, ["degree_decile"], seed)
        if len(n2):
            n2_decoys.add((n2.iloc[0]["protein_a"], n2.iloc[0]["protein_b"]))
        if len(n6):
            n6_decoys.add((n6.iloc[0]["protein_a"], n6.iloc[0]["protein_b"]))

    assert all("P1" in pair or "P2" in pair for pair in n2_decoys), \
        "every N2 decoy must anchor one endpoint at the source positive pair by construction"
    assert any("P1" not in pair and "P2" not in pair for pair in n6_decoys), \
        "N6 should be able to produce a decoy sharing no endpoint with its source positive pair"
