"""Bootstrap confidence intervals.

PPI pairs are not independent samples -- pairs sharing a protein share
whatever that protein's embedding/degree contributes to the prediction.
A naive pair-level bootstrap treats every row as independent and understates
the true uncertainty. This uses a connected-component (graph) bootstrap
instead: build the undirected graph whose edges are the test pairs, find its
connected components, and resample WHOLE COMPONENTS with replacement -- so
replicates vary at the level of transitively-connected clusters of proteins,
respecting dependency introduced through either endpoint, and any chain of
shared endpoints, at once.

History: an earlier version of this function grouped test pairs by
protein_a alone. Because pairs are canonicalized so protein_a < protein_b
lexicographically (src/data/canonicalize.py), protein_a is not a stable
per-protein role -- a given protein is protein_a only for the subset of its
partners with a lexicographically larger ID, and protein_b for the rest. A
protein_a-only grouping therefore scattered a protein's own edges across
many other proteins' groups instead of clustering them together, which
understated the very dependency the bootstrap was meant to capture.

A second version tried resampling the pool of proteins directly (union of
both endpoint columns) and including every pair touching a resampled
protein. That was abandoned: 122 of 328 proteins in a representative test
set (N2/Attention Fusion, R3) appear as both protein_a and protein_b, so
each such protein's rows would belong to TWO different resampled units at
once -- an overlapping-membership structure standard cluster-bootstrap
theory does not cover, and empirically it *narrowed* at least one CI
relative to the protein_a-only version, the opposite of what properly
accounting for more dependency should do. Connected components avoid this
entirely: every row belongs to exactly one component (a genuine partition,
since sharing either endpoint places two rows in the same component
transitively), so standard cluster-bootstrap theory applies cleanly, and
the full transitive dependency chain -- not just direct shared endpoints --
is respected.

Once components were computed correctly, a further problem surfaced: this
benchmark's R3 test sets are each dominated by ONE giant connected
component (94.6% of N0's 462 pairs; 98.2% of N2's 446 pairs), leaving only
5-19 components total, nearly all of them negligibly small. This is the
classic "few clusters" regime in the cluster-robust-inference literature,
where a percentile cluster bootstrap (resampling G cluster keys with
replacement, G very small and wildly unequal in size) is known to produce
erratic, unreliable intervals -- which is exactly what was observed
(implausibly wide or narrow CIs depending on how often the giant component
happened to be drawn). The functions below therefore use a delete-one-
cluster JACKKNIFE with a t(G-1) reference distribution instead of a
percentile bootstrap for the final confidence intervals: this is a
standard remedy for small-G cluster-robust inference (the "jackknife
repeated replication" approach from complex-survey statistics; see also
Cameron & Miller, 2015, "A Practitioner's Guide to Cluster-Robust
Inference"), works with as few as 2 clusters, and does not require
comparable cluster sizes -- unlike the percentile bootstrap, it does not
depend on how often any one cluster happens to be resampled, only on the
G leave-one-cluster-out point estimates themselves.
"""
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg step-up FDR correction for a family of hypothesis
    tests conducted together (Section 3.8 runs many models x negative sets
    x metrics x CI methods, and a per-test 95% CI alone does not control
    the family-wise false-discovery rate across that many comparisons).
    Returns (reject, adjusted_p_value) per input p-value, same order as
    `p_values`. NaN p-values (e.g. from an undefined jackknife CI) are
    excluded from the correction and returned as not-rejected/NaN."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    reject = np.zeros(len(p), dtype=bool)
    adjusted = np.full(len(p), np.nan)
    if m == 0:
        return reject.tolist(), adjusted.tolist()

    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid_idx])]
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    raw_adj = ranked * m / ranks
    adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    adjusted[order] = adj_sorted

    thresholds = ranks / m * alpha
    below = ranked <= thresholds
    if below.any():
        k_max = np.max(np.where(below)[0])
        reject[order[:k_max + 1]] = True
    return reject.tolist(), adjusted.tolist()


def connected_component_groups(pairs: pd.DataFrame) -> dict[int, np.ndarray]:
    """component_id -> array of row-positions into `pairs` whose edge
    (protein_a, protein_b) lies in that connected component of the
    undirected graph formed by all rows."""
    g = nx.Graph()
    g.add_nodes_from(pd.concat([pairs["protein_a"], pairs["protein_b"]]).unique())
    g.add_edges_from(zip(pairs["protein_a"], pairs["protein_b"]))

    protein_to_component = {}
    for component_id, component_nodes in enumerate(nx.connected_components(g)):
        for node in component_nodes:
            protein_to_component[node] = component_id

    row_component = pairs["protein_a"].map(protein_to_component).to_numpy()
    groups: dict[int, list[int]] = {}
    for row, comp in enumerate(row_component):
        groups.setdefault(comp, []).append(row)
    return {comp: np.array(rows) for comp, rows in groups.items()}


def protein_cluster_bootstrap_indices(pairs: pd.DataFrame, n_boot: int, seed: int) -> list[np.ndarray]:
    """Returns n_boot arrays of row-positions into `pairs` (0-indexed,
    aligned with reset_index(drop=True)), each a connected-component (graph)
    resample: whole components (proteins transitively linked via any chain
    of shared endpoints, plus every row touching them) are resampled with
    replacement."""
    pairs = pairs.reset_index(drop=True)
    groups = connected_component_groups(pairs)
    group_keys = list(groups.keys())
    rng = np.random.default_rng(seed)

    replicates = []
    for _ in range(n_boot):
        sampled_keys = rng.choice(group_keys, size=len(group_keys), replace=True)
        idx = np.concatenate([groups[k] for k in sampled_keys])
        replicates.append(idx)
    return replicates


def _empirical_two_sided_p(values: np.ndarray, null: float) -> float:
    """Two-sided empirical bootstrap p-value: twice the smaller tail
    fraction of the bootstrap/jackknife distribution on either side of
    `null` (the standard construction for a percentile-bootstrap
    hypothesis test; see e.g. Davison & Hinkley, 1997, Sec. 4.4)."""
    p_le = np.mean(values <= null)
    p_ge = np.mean(values >= null)
    return float(min(1.0, 2 * min(p_le, p_ge)))


def bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, bootstrap_indices: list[np.ndarray],
    metric_fn, ci: float = 0.95, null: float = 0.5,
) -> dict:
    """Point estimate on the full sample; percentile CI from the bootstrap
    distribution. `metric_fn(y_true, y_prob) -> float`. `null` is the
    chance/no-effect reference value used only for the two-sided p-value
    (0.5 for ROC-AUC, 0.0 for MCC) -- the CI itself does not depend on it."""
    point = metric_fn(y_true, y_prob)
    boot_values = []
    for idx in bootstrap_indices:
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue  # metric undefined (e.g. ROC-AUC) for a single-class resample
        boot_values.append(metric_fn(yt, yp))
    boot_values = np.array(boot_values)
    alpha = (1 - ci) / 2
    lower, upper = np.percentile(boot_values, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point": float(point), "ci_low": float(lower), "ci_high": float(upper),
        "n_valid_replicates": len(boot_values),
        "p_value": _empirical_two_sided_p(boot_values, null),
    }


def paired_bootstrap_diff_ci(
    y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray,
    bootstrap_indices: list[np.ndarray], metric_fn, ci: float = 0.95,
) -> dict:
    """For two models scored on the SAME test set (same y_true, same row
    alignment): CI on metric(A) - metric(B), using the same resampled
    indices for both so within-replicate variance cancels correctly."""
    point = metric_fn(y_true, y_prob_a) - metric_fn(y_true, y_prob_b)
    diffs = []
    for idx in bootstrap_indices:
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(metric_fn(yt, y_prob_a[idx]) - metric_fn(yt, y_prob_b[idx]))
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    lower, upper = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point": float(point), "ci_low": float(lower), "ci_high": float(upper),
        "n_valid_replicates": len(diffs),
        "significant": bool(lower > 0 or upper < 0),
        "p_value": _empirical_two_sided_p(diffs, 0.0),
    }


def _jackknife_variance(loo_values: np.ndarray) -> float:
    """Delete-1-cluster jackknife variance (Wolter's JRR formula):
    ((G-1)/G) * sum((theta_hat_(-g) - mean(theta_hat_(-g)))^2), where
    theta_hat_(-g) is the statistic recomputed with cluster g removed.
    Does not require equal cluster sizes."""
    g = len(loo_values)
    return ((g - 1) / g) * np.sum((loo_values - loo_values.mean()) ** 2)


def _jackknife_p_value(point: float, se: float, df: int, null: float) -> float:
    """Two-sided p-value from a t(df) reference distribution for the
    jackknife point estimate against `null`. If se == 0 (every
    leave-one-cluster-out replicate identical), the t-statistic is
    degenerate: p is 0 if point differs from null (arbitrarily small
    variance would make any such difference significant) and 1 if it
    exactly equals null."""
    if se == 0:
        return 0.0 if point != null else 1.0
    t_stat = (point - null) / se
    return float(2 * stats.t.sf(abs(t_stat), df=df))


def leave_one_component_out_table(y_true: np.ndarray, y_prob: np.ndarray, groups: dict, metric_fn) -> pd.DataFrame:
    """One row per connected component: the metric recomputed with that
    component's rows excluded (the delete-1-cluster jackknife's internal
    leave-one-out step, exposed directly rather than only summarized into
    a variance estimate). This is the "leave-one-connected-component-out
    sensitivity analysis": does the effect survive removing any one component, especially
    the giant component that dominates this benchmark's R3 test sets?
    `groups` is a component_id -> row-index-array mapping, e.g. from
    `connected_component_groups`. A row is NaN in `loo_metric` if removing
    that component leaves a single-class remainder, for which `metric_fn`
    (e.g. ROC-AUC) is undefined -- reported as NaN rather than dropped, so
    the table's row count always equals the component count."""
    all_idx = np.arange(len(y_true))
    sizes = {comp: len(rows) for comp, rows in groups.items()}
    giant = max(sizes, key=sizes.get) if sizes else None

    rows = []
    for comp, comp_rows in groups.items():
        keep = np.setdiff1d(all_idx, comp_rows, assume_unique=True)
        yt = y_true[keep]
        loo_metric = metric_fn(yt, y_prob[keep]) if len(np.unique(yt)) >= 2 else float("nan")
        rows.append({
            "component_id": comp, "n_rows_in_component": sizes[comp],
            "n_rows_remaining": len(keep), "is_giant_component": comp == giant,
            "loo_metric": loo_metric,
        })
    return pd.DataFrame(rows)


def cluster_jackknife_ci(
    y_true: np.ndarray, y_prob: np.ndarray, groups: dict, metric_fn, ci: float = 0.95, null: float = 0.5,
) -> dict:
    """Delete-1-cluster jackknife CI with a t(G_eff - 1) reference
    distribution -- the small-G-appropriate replacement for a percentile
    cluster bootstrap when the number of connected components is small
    and/or wildly unequal in size (see module docstring). `groups` is a
    component_id -> row-index-array mapping, e.g. from
    `connected_component_groups`. A component is skipped (G_eff reduced by
    one) if removing it leaves a single-class remainder, for which
    `metric_fn` (e.g. ROC-AUC) is undefined. `null` is the chance/no-effect
    reference value used only for the two-sided p-value (0.5 for ROC-AUC,
    0.0 for MCC) -- the CI itself does not depend on it."""
    point = metric_fn(y_true, y_prob)
    loo_values = leave_one_component_out_table(y_true, y_prob, groups, metric_fn)["loo_metric"].dropna().to_numpy()
    g_eff = len(loo_values)

    if g_eff < 2:
        return {"point": float(point), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_clusters": len(groups), "n_clusters_used": g_eff, "significant": False,
                "p_value": float("nan"),
                "note": "fewer than 2 usable clusters after leave-one-out; jackknife CI undefined"}

    var = _jackknife_variance(loo_values)
    se = np.sqrt(var)
    t_crit = stats.t.ppf(1 - (1 - ci) / 2, df=g_eff - 1)
    lo, hi = point - t_crit * se, point + t_crit * se
    return {
        "point": float(point), "ci_low": float(lo), "ci_high": float(hi),
        "n_clusters": len(groups), "n_clusters_used": g_eff,
        "se": float(se), "df": g_eff - 1,
        "p_value": _jackknife_p_value(point, se, g_eff - 1, null),
    }


def paired_cluster_jackknife_diff_ci(
    y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray,
    groups: dict, metric_fn, ci: float = 0.95,
) -> dict:
    """Jackknife CI on metric(A) - metric(B) for two models scored on the
    SAME test set (same clusters), analogous to `paired_bootstrap_diff_ci`
    but using the delete-1-cluster jackknife instead of a percentile
    bootstrap."""
    point = metric_fn(y_true, y_prob_a) - metric_fn(y_true, y_prob_b)
    all_idx = np.arange(len(y_true))

    loo_values = []
    for comp_rows in groups.values():
        keep = np.setdiff1d(all_idx, comp_rows, assume_unique=True)
        yt = y_true[keep]
        if len(np.unique(yt)) < 2:
            continue
        loo_values.append(metric_fn(yt, y_prob_a[keep]) - metric_fn(yt, y_prob_b[keep]))
    loo_values = np.array(loo_values)
    g_eff = len(loo_values)

    if g_eff < 2:
        return {"point": float(point), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_clusters": len(groups), "n_clusters_used": g_eff, "significant": False,
                "p_value": float("nan"),
                "note": "fewer than 2 usable clusters after leave-one-out; jackknife CI undefined"}

    var = _jackknife_variance(loo_values)
    se = np.sqrt(var)
    t_crit = stats.t.ppf(1 - (1 - ci) / 2, df=g_eff - 1)
    lo, hi = point - t_crit * se, point + t_crit * se
    return {
        "point": float(point), "ci_low": float(lo), "ci_high": float(hi),
        "n_clusters": len(groups), "n_clusters_used": g_eff, "df": g_eff - 1,
        "significant": bool(lo > 0 or hi < 0),
        "p_value": _jackknife_p_value(point, se, g_eff - 1, 0.0),
    }


def unpaired_cluster_jackknife_diff_ci(
    y_true_a: np.ndarray, y_prob_a: np.ndarray, groups_a: dict,
    y_true_b: np.ndarray, y_prob_b: np.ndarray, groups_b: dict,
    metric_fn, ci: float = 0.95,
) -> dict:
    """Jackknife CI on metric(A) - metric(B) for two models scored on
    DIFFERENT, non-overlapping test sets (e.g. N0 vs. N2), each with its
    own connected-component structure. Combines each side's jackknife
    variance as independent samples (Var(A-B) = Var(A) + Var(B)) with a
    Welch-Satterthwaite degrees-of-freedom approximation for the
    reference t-distribution -- the standard way to combine two
    cluster-robust variance estimates from independent samples."""
    ci_a = cluster_jackknife_ci(y_true_a, y_prob_a, groups_a, metric_fn, ci)
    ci_b = cluster_jackknife_ci(y_true_b, y_prob_b, groups_b, metric_fn, ci)
    point = ci_a["point"] - ci_b["point"]

    if ci_a["n_clusters_used"] < 2 or ci_b["n_clusters_used"] < 2:
        return {"point": float(point), "ci_low": float("nan"), "ci_high": float("nan"),
                "significant": False, "p_value": float("nan"),
                "note": "fewer than 2 usable clusters on at least one side; jackknife CI undefined"}

    var_a, var_b = ci_a["se"] ** 2, ci_b["se"] ** 2
    df_a, df_b = ci_a["df"], ci_b["df"]
    var_sum = var_a + var_b
    df = (var_sum ** 2) / ((var_a ** 2) / df_a + (var_b ** 2) / df_b)
    se = np.sqrt(var_sum)
    t_crit = stats.t.ppf(1 - (1 - ci) / 2, df=df)
    lo, hi = point - t_crit * se, point + t_crit * se
    return {
        "point": float(point), "ci_low": float(lo), "ci_high": float(hi),
        "df": float(df), "significant": bool(lo > 0 or hi < 0),
        "p_value": _jackknife_p_value(point, se, df, 0.0),
    }
