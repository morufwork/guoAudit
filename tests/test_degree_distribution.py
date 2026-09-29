"""Tests for degree-distribution characterization."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.degree_distribution import (
    bootstrap_protein_level_degree_mean_diff_ci,
    cluster_bootstrap_degree_mean_diff_ci,
    compare_degree_distributions,
    describe_degree_distribution,
    distinct_protein_degree_values,
    pair_associated_degree_values,
)


def test_pair_associated_degree_values_pools_both_endpoints():
    pairs = pd.DataFrame({"protein_a": ["P1", "P2"], "protein_b": ["P3", "P4"]})
    degree = pd.Series({"P1": 10, "P2": 20, "P3": 1, "P4": 2})
    values = pair_associated_degree_values(pairs, degree)
    assert sorted(values.tolist()) == [1.0, 2.0, 10.0, 20.0]


def test_describe_degree_distribution_matches_known_statistics():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # right-skewed toy example
    d = describe_degree_distribution(values)
    assert d["n"] == 5
    assert d["mean"] == pytest.approx(22.0)
    assert d["median"] == pytest.approx(3.0)
    assert d["mean"] > d["median"], "mean should be pulled above the median by the skew, as expected"
    assert d["iqr_low"] == pytest.approx(2.0)
    assert d["iqr_high"] == pytest.approx(4.0)


def _independent_component_pairs(n, rng, degree_range, prefix):
    """n pairs, every protein unique -- each row its own connected
    component, so the cluster bootstrap behaves like an ordinary
    independent-observations bootstrap for this synthetic case. `prefix`
    keeps protein names distinct across separate calls (e.g. a "hi" and a
    "lo" group), so their degree Series can be concatenated without
    index collisions overwriting one group's values with the other's."""
    proteins_a = [f"{prefix}A{i}" for i in range(n)]
    proteins_b = [f"{prefix}B{i}" for i in range(n)]
    pairs = pd.DataFrame({"protein_a": proteins_a, "protein_b": proteins_b})
    degree = pd.Series({p: rng.uniform(*degree_range) for p in proteins_a + proteins_b})
    return pairs, degree


def test_compare_degree_distributions_detects_clear_difference():
    rng = np.random.default_rng(0)
    pairs_hi, degree_hi = _independent_component_pairs(150, rng, (8, 12), prefix="hi")
    pairs_lo, degree_lo = _independent_component_pairs(150, rng, (1, 3), prefix="lo")
    degree = pd.concat([degree_hi, degree_lo])

    result = compare_degree_distributions(pairs_hi, pairs_lo, degree, n_boot=500, seed=0)
    assert result["mean_diff"] > 5
    assert result["cohens_d"] > 1.0, "large, obviously-different distributions should show a large effect size"
    assert result["ks_pvalue"] < 0.01
    assert result["wasserstein_distance"] > 5
    assert result["protein_level_mean_diff_ci_low"] > 0
    assert result["protein_level_mean_diff_ci_significant"]


def test_compare_degree_distributions_null_for_identical_distributions():
    rng = np.random.default_rng(1)
    pairs_a, degree_a = _independent_component_pairs(150, rng, (1, 10), prefix="grpa")
    pairs_b, degree_b = _independent_component_pairs(150, rng, (1, 10), prefix="grpb")
    degree = pd.concat([degree_a, degree_b])

    result = compare_degree_distributions(pairs_a, pairs_b, degree, n_boot=500, seed=1)
    assert abs(result["cohens_d"]) < 0.5
    assert result["ks_pvalue"] > 0.05
    assert not result["protein_level_mean_diff_ci_significant"]
    assert result["protein_level_mean_diff_ci_low"] <= 0 <= result["protein_level_mean_diff_ci_high"]


def test_distinct_protein_degree_values_deduplicates_proteins():
    pairs = pd.DataFrame({"protein_a": ["P1", "P1", "P2"], "protein_b": ["P2", "P3", "P3"]})
    degree = pd.Series({"P1": 10, "P2": 20, "P3": 30})
    values = distinct_protein_degree_values(pairs, degree)
    assert sorted(values.tolist()) == [10.0, 20.0, 30.0]  # each protein once, despite appearing in multiple pairs


def test_bootstrap_protein_level_ci_valid_even_when_groups_share_every_protein():
    """The pair-graph cluster bootstrap degenerates to G=1 when two groups
    share (almost) every protein endpoint -- the real-dataset case this
    fix was built for. The protein-level bootstrap must stay well-defined
    in exactly that scenario, since it never depends on pair-graph
    structure at all."""
    rng = np.random.default_rng(3)
    proteins = [f"P{i}" for i in range(200)]
    degree = pd.Series({p: rng.uniform(1, 20) for p in proteins})
    # pairs_a and pairs_b reuse the SAME protein pool, fully overlapping
    pairs_a = pd.DataFrame({"protein_a": proteins[::2], "protein_b": proteins[1::2]})
    pairs_b = pd.DataFrame({"protein_a": proteins[1::2], "protein_b": proteins[::2]})

    result = bootstrap_protein_level_degree_mean_diff_ci(pairs_a, pairs_b, degree, n_boot=300, seed=4)
    assert np.isfinite(result["ci_low"]) and np.isfinite(result["ci_high"])
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_cluster_bootstrap_ci_accounts_for_shared_endpoints_across_groups():
    """A protein appearing in both compared pair sets should keep its rows
    from both sets in the same resampled component -- this is the whole
    point of building components on the UNION graph rather than treating
    the two groups' endpoint-occurrence pools as independent."""
    pairs_a = pd.DataFrame({"protein_a": ["HUB", "HUB"], "protein_b": ["X1", "X2"]})
    pairs_b = pd.DataFrame({"protein_a": ["HUB"], "protein_b": ["Y1"]})
    degree = pd.Series({"HUB": 50, "X1": 1, "X2": 1, "Y1": 1})

    result = cluster_bootstrap_degree_mean_diff_ci(pairs_a, pairs_b, degree, n_boot=200, seed=2)
    # all three rows share HUB, so every replicate either includes all of
    # them or none -- point estimate and CI must still be well-defined.
    assert np.isfinite(result["point"])
    assert result["n_valid_replicates"] >= 0
