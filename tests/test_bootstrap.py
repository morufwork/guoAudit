"""Bootstrap CI correctness."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from src.evaluation.bootstrap import (
    benjamini_hochberg,
    bootstrap_ci,
    cluster_jackknife_ci,
    connected_component_groups,
    leave_one_component_out_table,
    paired_bootstrap_diff_ci,
    paired_cluster_jackknife_diff_ci,
    protein_cluster_bootstrap_indices,
    unpaired_cluster_jackknife_diff_ci,
)


@pytest.fixture
def toy_pairs():
    # Two disjoint connected components: {P1,P2,Q1,Q2} (rows 0-2) and
    # {P3,P4,Q3,Q4} (rows 3-5), each a 3-edge component.
    return pd.DataFrame({
        "protein_a": ["P1", "P1", "P2", "P3", "P3", "P4"],
        "protein_b": ["Q1", "Q2", "Q1", "Q3", "Q4", "Q3"],
        "label": [1, 0, 1, 0, 1, 0],
    })


def test_protein_cluster_bootstrap_groups_by_connected_component(toy_pairs):
    idx = protein_cluster_bootstrap_indices(toy_pairs, n_boot=100, seed=0)
    assert len(idx) == 100
    component_1, component_2 = {0, 1, 2}, {3, 4, 5}
    for rep in idx:
        # every replicate must be a whole-component block: each row present
        # must bring its entire component along, never a partial component.
        rep_set = set(rep.tolist())
        assert rep_set.issubset(component_1 | component_2)
        if rep_set & component_1:
            assert component_1.issubset(rep_set)
        if rep_set & component_2:
            assert component_2.issubset(rep_set)


def test_protein_cluster_bootstrap_connects_shared_endpoint_across_columns():
    """Regression test: two rows sharing "HUB" as an
    endpoint -- once as protein_b (row 0), once as protein_a (row 1) --
    must be treated as ONE dependency cluster. A protein_a-only grouping
    (the pre-fix bug) would wrongly separate them: row 0's only key is P1
    (HUB is invisible there, since it's protein_b); row 1's key is HUB.
    The connected-component bootstrap must keep both rows together in every
    replicate, since P1-HUB-P2 is a single connected chain."""
    pairs = pd.DataFrame({
        "protein_a": ["P1", "HUB"],
        "protein_b": ["HUB", "P2"],
        "label": [1, 0],
    })
    idx = protein_cluster_bootstrap_indices(pairs, n_boot=200, seed=0)
    for rep in idx:
        assert set(rep.tolist()) == {0, 1}, "both rows share the connected component and must always co-occur"


def test_protein_cluster_bootstrap_separates_independent_components():
    """Two fully disjoint components (no shared protein at all) must vary
    independently across replicates -- confirms components aren't merged
    when they share nothing."""
    pairs = pd.DataFrame({
        "protein_a": ["P1", "P3"],
        "protein_b": ["P2", "P4"],
        "label": [1, 0],
    })
    idx = protein_cluster_bootstrap_indices(pairs, n_boot=200, seed=0)
    signatures = {tuple(sorted(set(rep.tolist()))) for rep in idx}
    # possible outcomes: {0}, {1}, {0,1}, or (rare) neither -- more than one
    # distinct signature confirms the two components resample independently
    assert len(signatures) > 1


def test_protein_cluster_bootstrap_is_seed_reproducible(toy_pairs):
    idx1 = protein_cluster_bootstrap_indices(toy_pairs, n_boot=50, seed=42)
    idx2 = protein_cluster_bootstrap_indices(toy_pairs, n_boot=50, seed=42)
    for a, b in zip(idx1, idx2):
        np.testing.assert_array_equal(a, b)


def test_bootstrap_ci_point_matches_direct_computation():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.3, size=n), 0, 1)
    pairs = pd.DataFrame({"protein_a": [f"P{i}" for i in range(n)], "protein_b": [f"Q{i}" for i in range(n)], "label": y_true})

    idx = protein_cluster_bootstrap_indices(pairs, n_boot=500, seed=1)
    result = bootstrap_ci(y_true, y_prob, idx, roc_auc_score)
    assert result["point"] == pytest.approx(roc_auc_score(y_true, y_prob))
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_bootstrap_ci_detects_clear_chance_vs_signal():
    rng = np.random.default_rng(0)
    n = 300
    # strong signal
    y_true_signal = rng.integers(0, 2, size=n)
    y_prob_signal = np.clip(y_true_signal * 0.9 + rng.normal(0, 0.1, size=n), 0, 1)
    # pure noise
    y_true_noise = rng.integers(0, 2, size=n)
    y_prob_noise = rng.uniform(0, 1, size=n)

    pairs = pd.DataFrame({"protein_a": [f"P{i}" for i in range(n)], "protein_b": [f"Q{i}" for i in range(n)]})
    idx = protein_cluster_bootstrap_indices(pairs, n_boot=500, seed=2)

    signal_ci = bootstrap_ci(y_true_signal, y_prob_signal, idx, roc_auc_score)
    noise_ci = bootstrap_ci(y_true_noise, y_prob_noise, idx, roc_auc_score)

    assert signal_ci["ci_low"] > 0.5
    assert noise_ci["ci_low"] < 0.5 < noise_ci["ci_high"]


def test_paired_bootstrap_diff_ci_zero_for_identical_predictions():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.clip(y_true * 0.5 + rng.normal(0, 0.3, size=n), 0, 1)
    pairs = pd.DataFrame({"protein_a": [f"P{i}" for i in range(n)], "protein_b": [f"Q{i}" for i in range(n)]})
    idx = protein_cluster_bootstrap_indices(pairs, n_boot=300, seed=3)

    result = paired_bootstrap_diff_ci(y_true, y_prob, y_prob, idx, roc_auc_score)
    assert result["point"] == 0.0
    assert not result["significant"]
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_paired_bootstrap_diff_ci_detects_clear_difference():
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.integers(0, 2, size=n)
    y_prob_good = np.clip(y_true * 0.9 + rng.normal(0, 0.1, size=n), 0, 1)
    y_prob_bad = rng.uniform(0, 1, size=n)
    pairs = pd.DataFrame({"protein_a": [f"P{i}" for i in range(n)], "protein_b": [f"Q{i}" for i in range(n)]})
    idx = protein_cluster_bootstrap_indices(pairs, n_boot=500, seed=4)

    result = paired_bootstrap_diff_ci(y_true, y_prob_good, y_prob_bad, idx, roc_auc_score)
    assert result["significant"]
    assert result["ci_low"] > 0


# --- Cluster jackknife ---

def _independent_pairs(n, rng, y_true=None):
    """n rows, every protein unique -- each row is its own connected component."""
    if y_true is None:
        y_true = rng.integers(0, 2, size=n)
    return pd.DataFrame({"protein_a": [f"P{i}" for i in range(n)], "protein_b": [f"Q{i}" for i in range(n)]}), y_true


def test_cluster_jackknife_ci_detects_clear_signal():
    rng = np.random.default_rng(0)
    n = 300
    pairs, y_true = _independent_pairs(n, rng)
    y_prob = np.clip(y_true * 0.9 + rng.normal(0, 0.1, size=n), 0, 1)
    groups = connected_component_groups(pairs)

    result = cluster_jackknife_ci(y_true, y_prob, groups, roc_auc_score)
    assert result["n_clusters"] == n  # every row its own component here
    assert result["ci_low"] > 0.5


def test_cluster_jackknife_ci_detects_chance():
    rng = np.random.default_rng(1)
    n = 300
    pairs, y_true = _independent_pairs(n, rng)
    y_prob = rng.uniform(0, 1, size=n)
    groups = connected_component_groups(pairs)

    result = cluster_jackknife_ci(y_true, y_prob, groups, roc_auc_score)
    assert result["ci_low"] < 0.5 < result["ci_high"]


def test_cluster_jackknife_ci_undefined_with_fewer_than_two_usable_clusters():
    # a single connected component: removing it leaves nothing to compute
    # a leave-one-out estimate from -- g_eff must end up < 2.
    pairs = pd.DataFrame({"protein_a": ["P1", "P2"], "protein_b": ["Q1", "P1"]})
    y_true = np.array([1, 0])
    y_prob = np.array([0.8, 0.2])
    groups = connected_component_groups(pairs)  # single component (P1-Q1-P2 chain)

    result = cluster_jackknife_ci(y_true, y_prob, groups, roc_auc_score)
    assert result["n_clusters_used"] < 2
    assert np.isnan(result["ci_low"]) and np.isnan(result["ci_high"])


def test_cluster_jackknife_ci_wide_under_one_dominant_component():
    """Regression test mirroring the real finding: when
    one connected component holds nearly all the data and a few tiny
    components make up the rest, the jackknife CI must be wide (few usable
    clusters -> large t-quantile), not spuriously tight the way a naive
    percentile bootstrap over the same lopsided components was observed to
    be."""
    rng = np.random.default_rng(2)
    n_giant = 190
    # one giant component: a path graph P0-P1-...-P_{n_giant} (all one component)
    giant_a = [f"G{i}" for i in range(n_giant)]
    giant_b = [f"G{i+1}" for i in range(n_giant)]
    # plus a handful of small, separate 2-node components
    small_a = [f"S{i}a" for i in range(6)]
    small_b = [f"S{i}b" for i in range(6)]

    pairs = pd.DataFrame({"protein_a": giant_a + small_a, "protein_b": giant_b + small_b})
    y_true = rng.integers(0, 2, size=len(pairs))
    # noise std wide enough that AUC < 1 (imperfect separation) -- otherwise
    # every leave-one-component-out estimate is also a perfect 1.0, giving
    # zero jackknife variance and defeating the point of this test.
    y_prob = np.clip(y_true * 0.9 + rng.normal(0, 0.35, size=len(pairs)), 0, 1)
    groups = connected_component_groups(pairs)
    assert len(groups) == 7  # 1 giant + 6 small

    result = cluster_jackknife_ci(y_true, y_prob, groups, roc_auc_score)
    assert result["ci_high"] - result["ci_low"] > 0.3  # wide: only ~7 usable clusters


def test_paired_cluster_jackknife_diff_ci_zero_for_identical_predictions():
    rng = np.random.default_rng(3)
    n = 200
    pairs, y_true = _independent_pairs(n, rng)
    y_prob = np.clip(y_true * 0.5 + rng.normal(0, 0.3, size=n), 0, 1)
    groups = connected_component_groups(pairs)

    result = paired_cluster_jackknife_diff_ci(y_true, y_prob, y_prob, groups, roc_auc_score)
    assert result["point"] == 0.0
    assert not result["significant"]
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_paired_cluster_jackknife_diff_ci_detects_clear_difference():
    rng = np.random.default_rng(4)
    n = 300
    pairs, y_true = _independent_pairs(n, rng)
    y_prob_good = np.clip(y_true * 0.9 + rng.normal(0, 0.1, size=n), 0, 1)
    y_prob_bad = rng.uniform(0, 1, size=n)
    groups = connected_component_groups(pairs)

    result = paired_cluster_jackknife_diff_ci(y_true, y_prob_good, y_prob_bad, groups, roc_auc_score)
    assert result["significant"]
    assert result["ci_low"] > 0


def test_unpaired_cluster_jackknife_diff_ci_detects_clear_difference():
    rng = np.random.default_rng(5)
    n = 300
    pairs_a, y_true_a = _independent_pairs(n, rng)
    y_prob_a = np.clip(y_true_a * 0.9 + rng.normal(0, 0.1, size=n), 0, 1)
    groups_a = connected_component_groups(pairs_a)

    pairs_b, y_true_b = _independent_pairs(n, rng)
    y_prob_b = rng.uniform(0, 1, size=n)
    groups_b = connected_component_groups(pairs_b)

    result = unpaired_cluster_jackknife_diff_ci(y_true_a, y_prob_a, groups_a, y_true_b, y_prob_b, groups_b, roc_auc_score)
    assert result["significant"]
    assert result["ci_low"] > 0
    assert result["df"] > 0


def test_unpaired_cluster_jackknife_diff_ci_zero_for_identical_distributions():
    rng = np.random.default_rng(6)
    n = 250
    pairs_a, y_true_a = _independent_pairs(n, rng)
    y_prob_a = np.clip(y_true_a * 0.7 + rng.normal(0, 0.2, size=n), 0, 1)
    groups_a = connected_component_groups(pairs_a)

    pairs_b, y_true_b = _independent_pairs(n, rng)
    y_prob_b = np.clip(y_true_b * 0.7 + rng.normal(0, 0.2, size=n), 0, 1)
    groups_b = connected_component_groups(pairs_b)

    result = unpaired_cluster_jackknife_diff_ci(y_true_a, y_prob_a, groups_a, y_true_b, y_prob_b, groups_b, roc_auc_score)
    assert not result["significant"]
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


# --- p-values ---

def test_bootstrap_ci_p_value_small_for_clear_signal_large_for_noise():
    rng = np.random.default_rng(10)
    n = 300
    pairs, y_true = _independent_pairs(n, rng)
    idx = protein_cluster_bootstrap_indices(pairs, n_boot=500, seed=0)

    y_prob_signal = np.clip(y_true * 0.9 + rng.normal(0, 0.1, size=n), 0, 1)
    result_signal = bootstrap_ci(y_true, y_prob_signal, idx, roc_auc_score, null=0.5)
    assert result_signal["p_value"] < 0.01

    y_prob_noise = rng.uniform(0, 1, size=n)
    result_noise = bootstrap_ci(y_true, y_prob_noise, idx, roc_auc_score, null=0.5)
    assert result_noise["p_value"] > 0.2


def test_cluster_jackknife_ci_p_value_small_for_clear_signal_large_for_noise():
    rng = np.random.default_rng(11)
    n = 300
    pairs, y_true = _independent_pairs(n, rng)
    groups = connected_component_groups(pairs)

    y_prob_signal = np.clip(y_true * 0.9 + rng.normal(0, 0.1, size=n), 0, 1)
    result_signal = cluster_jackknife_ci(y_true, y_prob_signal, groups, roc_auc_score, null=0.5)
    assert result_signal["p_value"] < 0.01

    y_prob_noise = rng.uniform(0, 1, size=n)
    result_noise = cluster_jackknife_ci(y_true, y_prob_noise, groups, roc_auc_score, null=0.5)
    assert result_noise["p_value"] > 0.2


def test_paired_and_unpaired_jackknife_p_value_present_and_consistent_with_significant_flag():
    rng = np.random.default_rng(12)
    n = 300
    pairs, y_true = _independent_pairs(n, rng)
    groups = connected_component_groups(pairs)
    y_prob_good = np.clip(y_true * 0.9 + rng.normal(0, 0.1, size=n), 0, 1)
    y_prob_bad = rng.uniform(0, 1, size=n)

    paired = paired_cluster_jackknife_diff_ci(y_true, y_prob_good, y_prob_bad, groups, roc_auc_score)
    assert paired["significant"]
    assert paired["p_value"] < 0.05

    pairs_b, y_true_b = _independent_pairs(n, rng)
    groups_b = connected_component_groups(pairs_b)
    y_prob_b = np.clip(y_true_b * 0.7 + rng.normal(0, 0.2, size=n), 0, 1)
    unpaired = unpaired_cluster_jackknife_diff_ci(y_true, y_prob_good, groups, y_true_b, y_prob_b, groups_b, roc_auc_score)
    assert unpaired["significant"]
    assert unpaired["p_value"] < 0.05


def test_benjamini_hochberg_matches_hand_computed_example():
    p_values = [0.001, 0.01, 0.02, 0.5]
    reject, adjusted = benjamini_hochberg(p_values, alpha=0.05)
    assert reject == [True, True, True, False]
    assert adjusted == pytest.approx([0.004, 0.02, 4 * 0.02 / 3, 0.5], abs=1e-9)


def test_benjamini_hochberg_rejects_none_when_all_p_values_large():
    p_values = [0.2, 0.3, 0.4, 0.9]
    reject, adjusted = benjamini_hochberg(p_values, alpha=0.05)
    assert reject == [False, False, False, False]


def test_benjamini_hochberg_excludes_nan_from_correction():
    p_values = [0.01, float("nan"), 0.2]
    reject, adjusted = benjamini_hochberg(p_values, alpha=0.05)
    assert reject == [True, False, False]
    assert adjusted[0] == pytest.approx(0.02)
    assert np.isnan(adjusted[1])
    assert adjusted[2] == pytest.approx(0.2)


def test_benjamini_hochberg_more_lenient_than_bonferroni():
    # BH should reject at least as many hypotheses as a Bonferroni correction
    # at the same alpha -- the whole point of using FDR control here instead
    # of the more conservative Bonferroni option.
    rng = np.random.default_rng(13)
    p_values = sorted(rng.uniform(0, 0.05, size=8).tolist() + rng.uniform(0.1, 1.0, size=12).tolist())
    reject_bh, _ = benjamini_hochberg(p_values, alpha=0.05)
    bonferroni_reject = [p <= 0.05 / len(p_values) for p in p_values]
    assert sum(reject_bh) >= sum(bonferroni_reject)


# --- Leave-one-connected-component-out sensitivity analysis ---

def test_leave_one_component_out_table_has_one_row_per_component(toy_pairs):
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3])
    groups = connected_component_groups(toy_pairs)

    table = leave_one_component_out_table(y_true, y_prob, groups, roc_auc_score)
    assert len(table) == len(groups) == 2
    assert set(table["component_id"]) == set(groups.keys())
    assert table["n_rows_in_component"].sum() == len(toy_pairs)


def test_leave_one_component_out_table_flags_exactly_one_giant_component():
    rng = np.random.default_rng(20)
    n_giant, n_small = 190, 6
    giant_a = [f"G{i}" for i in range(n_giant)]
    giant_b = [f"G{i + 1}" for i in range(n_giant)]
    small_a = [f"S{i}a" for i in range(n_small)]
    small_b = [f"S{i}b" for i in range(n_small)]
    pairs = pd.DataFrame({"protein_a": giant_a + small_a, "protein_b": giant_b + small_b})
    y_true = rng.integers(0, 2, size=len(pairs))
    y_prob = rng.uniform(0, 1, size=len(pairs))
    groups = connected_component_groups(pairs)

    table = leave_one_component_out_table(y_true, y_prob, groups, roc_auc_score)
    assert table["is_giant_component"].sum() == 1
    giant_row = table[table["is_giant_component"]].iloc[0]
    assert giant_row["n_rows_in_component"] == n_giant  # path graph: n_giant edges (rows) span n_giant+1 nodes
    # every non-giant component's leave-out removes almost nothing; the giant's removes almost everything
    assert table.loc[table["is_giant_component"], "n_rows_remaining"].iloc[0] < table.loc[~table["is_giant_component"], "n_rows_remaining"].iloc[0]


def test_leave_one_component_out_table_marks_undefined_metric_as_nan_not_dropped():
    """Removing a component can leave a single-class remainder, for which
    ROC-AUC is undefined -- the row must stay in the table (NaN), not
    silently disappear, so the row count always equals the component
    count (the property the manuscript's sensitivity table depends on)."""
    pairs = pd.DataFrame({"protein_a": ["P1", "P2"], "protein_b": ["Q1", "Q2"]})  # 2 disjoint components
    y_true = np.array([1, 1])  # single class overall -- removing either component still leaves one class
    y_prob = np.array([0.6, 0.4])
    groups = connected_component_groups(pairs)

    table = leave_one_component_out_table(y_true, y_prob, groups, roc_auc_score)
    assert len(table) == 2
    assert table["loo_metric"].isna().all()


def test_cluster_jackknife_ci_uses_leave_one_component_out_table_consistently():
    """Regression guard for the refactor: cluster_jackknife_ci's g_eff and
    point estimates must match what leave_one_component_out_table reports
    directly, since the CI now computes its loo_values from that table."""
    rng = np.random.default_rng(21)
    n = 300
    pairs, y_true = _independent_pairs(n, rng)
    y_prob = np.clip(y_true * 0.9 + rng.normal(0, 0.1, size=n), 0, 1)
    groups = connected_component_groups(pairs)

    table = leave_one_component_out_table(y_true, y_prob, groups, roc_auc_score)
    result = cluster_jackknife_ci(y_true, y_prob, groups, roc_auc_score)
    assert result["n_clusters"] == len(table)
    assert result["n_clusters_used"] == table["loo_metric"].notna().sum()
