"""Degree-distribution characterization.

Section 3.5's central diagnostic reports only the MEAN pair-associated
degree for each pair set (positives 6.72, N0 negatives 3.34, etc.).
Network degree distributions are typically right-skewed, so a mean
difference alone under-characterizes how two distributions actually
differ. This module adds a fuller characterization: median/IQR/SD, and a distributional-comparison suite (Kolmogorov-
Smirnov statistic, Wasserstein distance, Cohen's d) computed descriptively
on the pair-associated (endpoint-occurrence-pooled) values, the same
pooling the existing 6.72/3.34 statistic already uses.

A CI for the mean difference is a separate, inferential claim that needs a
valid resampling unit -- naively resampling endpoint occurrences would
repeat the exact pseudo-replication problem Section 2.8's cluster bootstrap
was built to fix. The natural fix (resample connected components of the
pair graph, as Section 2.8 does) turns out to be degenerate here: on the
full dataset, the positive and N0 pair sets share so many protein
endpoints (2,141 of 2,217 positive-endpoint proteins also appear in an N0
pair) that their combined graph is a SINGLE connected component -- there
is only one resampling unit, so a component-level bootstrap or jackknife
CI cannot be computed (G=1). Resampling the protein pool directly instead
of pair-graph components was tried and rejected for exactly this kind of
statistic elsewhere in this project (src/evaluation/bootstrap.py's module
docstring): a protein can appear in many pairs, so pairs would belong to
multiple overlapping resampled units, a structure standard bootstrap
theory does not cover. Instead, `bootstrap_protein_level_degree_mean_diff_ci`
computes the CI on a related but distinct, cleanly resamplable quantity:
the mean degree of the DISTINCT proteins appearing in each pair set (one
observation per protein, not per pair-endpoint occurrence), which is a
standard, valid two-sample nonparametric bootstrap with no graph-
dependency machinery required.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.bootstrap import protein_cluster_bootstrap_indices


def pair_associated_degree_values(pairs: pd.DataFrame, degree: pd.Series) -> np.ndarray:
    """Pools `degree` over every pair-endpoint occurrence in `pairs` (one
    value per endpoint per row, so a hub protein appearing in many pairs
    contributes many times) -- the same pooling Section 3.5's headline
    mean positive-degree statistic (6.72 / 3.34) already uses."""
    return pd.concat([pairs["protein_a"].map(degree), pairs["protein_b"].map(degree)]).to_numpy(dtype=float)


def describe_degree_distribution(values: np.ndarray) -> dict:
    """Mean, median, IQR, SD -- what a mean alone (Section 3.5's original
    diagnostic) does not convey for a right-skewed distribution."""
    q25, q75 = np.percentile(values, [25, 75])
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)),
        "iqr_low": float(q25),
        "iqr_high": float(q75),
        "iqr": float(q75 - q25),
    }


def _cohens_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    n_a, n_b = len(values_a), len(values_b)
    var_a, var_b = np.var(values_a, ddof=1), np.var(values_b, ddof=1)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return 0.0 if np.mean(values_a) == np.mean(values_b) else float("inf")
    return float((np.mean(values_a) - np.mean(values_b)) / pooled_sd)


def cluster_bootstrap_degree_mean_diff_ci(
    pairs_a: pd.DataFrame, pairs_b: pd.DataFrame, degree: pd.Series,
    n_boot: int, seed: int, ci: float = 0.95,
) -> dict:
    """CI for the difference in pair-associated mean degree between two
    pair sets, resampling whole connected components of the COMBINED
    (pairs_a UNION pairs_b) dependency graph -- not independent
    endpoint-occurrence draws, which would understate uncertainty exactly
    the way a naive pair-level bootstrap did before Section 2.8's fix.
    Pairs_a and pairs_b may share protein endpoints (e.g. a protein
    appearing in both a positive pair and an N0 negative pair); building
    components on the union graph correctly keeps such cross-set
    dependence in the same resampled unit."""
    combined = pd.concat([
        pairs_a[["protein_a", "protein_b"]].assign(_group="a"),
        pairs_b[["protein_a", "protein_b"]].assign(_group="b"),
    ], ignore_index=True)

    def group_mean(sub: pd.DataFrame) -> float:
        return float(pd.concat([sub["protein_a"].map(degree), sub["protein_b"].map(degree)]).mean())

    point = group_mean(combined[combined["_group"] == "a"]) - group_mean(combined[combined["_group"] == "b"])

    diffs = []
    for idx in protein_cluster_bootstrap_indices(combined, n_boot, seed):
        sub = combined.iloc[idx]
        sub_a, sub_b = sub[sub["_group"] == "a"], sub[sub["_group"] == "b"]
        if len(sub_a) == 0 or len(sub_b) == 0:
            continue
        diffs.append(group_mean(sub_a) - group_mean(sub_b))
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_valid_replicates": len(diffs)}


def distinct_protein_degree_values(pairs: pd.DataFrame, degree: pd.Series) -> np.ndarray:
    """One value per DISTINCT protein appearing in `pairs` (not one per
    pair-endpoint occurrence) -- the resampling unit for
    `bootstrap_protein_level_degree_mean_diff_ci`, chosen specifically
    because it needs no pair-graph dependency structure at all."""
    proteins = pd.unique(pd.concat([pairs["protein_a"], pairs["protein_b"]]))
    return degree.reindex(proteins).dropna().to_numpy(dtype=float)


def bootstrap_protein_level_degree_mean_diff_ci(
    pairs_a: pd.DataFrame, pairs_b: pd.DataFrame, degree: pd.Series,
    n_boot: int, seed: int, ci: float = 0.95,
) -> dict:
    """CI for the difference in mean degree between the DISTINCT proteins
    appearing in two pair sets. Each protein contributes exactly one
    observation, so an ordinary two-sample nonparametric bootstrap
    (independently resample each group's protein list with replacement)
    is directly valid -- see module docstring for why the pair-associated
    (endpoint-occurrence-pooled) statistic's own natural resampling unit
    (connected pair-graph components) is degenerate for this comparison."""
    values_a = distinct_protein_degree_values(pairs_a, degree)
    values_b = distinct_protein_degree_values(pairs_b, degree)
    point = float(np.mean(values_a) - np.mean(values_b))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        boot_a = rng.choice(values_a, size=len(values_a), replace=True)
        boot_b = rng.choice(values_b, size=len(values_b), replace=True)
        diffs[i] = boot_a.mean() - boot_b.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point": point, "ci_low": float(lo), "ci_high": float(hi),
        "n_proteins_a": len(values_a), "n_proteins_b": len(values_b),
    }


def compare_degree_distributions(
    pairs_a: pd.DataFrame, pairs_b: pd.DataFrame, degree: pd.Series,
    n_boot: int, seed: int,
) -> dict:
    """Full distributional-comparison suite for two pair sets' degree
    distributions. KS statistic, Wasserstein distance, and Cohen's d are
    computed descriptively on the pair-associated (endpoint-occurrence-
    pooled) values -- the same pooling the existing mean diagnostic uses,
    reported as shape/effect-size descriptors rather than inferential
    claims requiring independent draws. The CI on the mean difference,
    which IS an inferential claim, is instead computed on the distinct-
    protein-level degree values (see `bootstrap_protein_level_degree_mean_diff_ci`
    and the module docstring for why)."""
    values_a = pair_associated_degree_values(pairs_a, degree)
    values_b = pair_associated_degree_values(pairs_b, degree)
    ks = stats.ks_2samp(values_a, values_b)
    protein_ci = bootstrap_protein_level_degree_mean_diff_ci(pairs_a, pairs_b, degree, n_boot, seed)
    # The pooled-SD Cohen's d above IS the standardized
    # mean difference (SMD) convention used in the propensity-score-matching
    # literature (Austin, 2011, Stat Med) to diagnose covariate balance after
    # matching -- same formula, different name/purpose. Reported again here
    # under that name, with the literature's own balance thresholds (|SMD| <
    # 0.1 "well balanced", < 0.25 "acceptable"), so a reader searching for
    # "SMD" specifically finds an explicit answer rather than having to
    # recognize Cohen's d as the same quantity.
    smd = _cohens_d(values_a, values_b)
    return {
        "mean_diff": float(np.mean(values_a) - np.mean(values_b)),
        "median_diff": float(np.median(values_a) - np.median(values_b)),
        "cohens_d": smd,
        "standardized_mean_difference": smd,
        "smd_well_balanced": bool(abs(smd) < 0.1),
        "smd_acceptable": bool(abs(smd) < 0.25),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein_distance": float(stats.wasserstein_distance(values_a, values_b)),
        "protein_level_mean_diff": protein_ci["point"],
        "protein_level_mean_diff_ci_low": protein_ci["ci_low"],
        "protein_level_mean_diff_ci_high": protein_ci["ci_high"],
        "protein_level_mean_diff_ci_significant": bool(protein_ci["ci_low"] > 0 or protein_ci["ci_high"] < 0),
        "n_proteins_a": protein_ci["n_proteins_a"],
        "n_proteins_b": protein_ci["n_proteins_b"],
    }
