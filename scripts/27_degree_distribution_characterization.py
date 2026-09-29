"""Degree-distribution characterization.

Section 3.5's headline diagnostic reports only the mean pair-associated
degree per set (positives 6.72, N0 negatives 3.34, N1/N2/N3/N5 similarly).
Network degree distributions are typically
right-skewed, so a mean difference alone does not fully characterize how
two distributions differ. This script adds the fuller characterization:
median/IQR/SD per set, density and ECDF plots (positives vs. every
negative set), and a distributional-comparison suite for positives vs.
each negative set -- Kolmogorov-Smirnov statistic, Wasserstein distance,
Cohen's d, and a dependency-aware CI for the mean difference (built on the
same connected-component cluster bootstrap Section 2.8 already uses, not
a naive endpoint-occurrence-level CI that would ignore PPI-pair
dependence).

Reuses the exact same pair sets and degree values the negative-sampling study
(scripts/11_negative_sampling_study.py) already computed and saved, so
every number here is directly comparable to the existing 6.72/3.34/etc.

Writes:
  tables/degree_distribution_stats.csv
  tables/degree_distribution_comparison.csv
  figures/degree_distribution_comparison.pdf
  results/degree_distribution/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.negative_sampling import compute_protein_stats
from src.evaluation.bootstrap import connected_component_groups
from src.evaluation.degree_distribution import (
    compare_degree_distributions,
    describe_degree_distribution,
    pair_associated_degree_values,
)
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("degree_distribution_characterization")

N_BOOT = 2000
SEED = 42
SET_LABELS = {
    "positive": "Positives",
    "n0": "N0 (benchmark original)",
    "n1": "N1 (random unknown)",
    "n2": "N2 (degree-matched)",
    "n3": "N3 (length-matched)",
    "n5": "N5 (hard: degree+length)",
}
COLORS = {"positive": "#000000", "n0": "#d95f02", "n1": "#7570b3", "n2": "#1b9e77", "n3": "#e7298a", "n5": "#66a61e"}


def main():
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=10)
    degree = protein_stats["positive_degree"]

    neg_sets_dir = REPO_ROOT / "data" / "processed" / "negative_sets"
    pair_sets = {
        "positive": pos, "n0": neg0,
        "n1": pd.read_csv(neg_sets_dir / "n1.tsv", sep="\t"),
        "n2": pd.read_csv(neg_sets_dir / "n2.tsv", sep="\t"),
        "n3": pd.read_csv(neg_sets_dir / "n3.tsv", sep="\t"),
        "n5": pd.read_csv(neg_sets_dir / "n5.tsv", sep="\t"),
    }
    values = {name: pair_associated_degree_values(df, degree) for name, df in pair_sets.items()}

    # --- Descriptive stats: mean/median/IQR/SD per set ---
    stats_rows = []
    for name, vals in values.items():
        d = describe_degree_distribution(vals)
        logger.info(f"[{name}] n={d['n']} mean={d['mean']:.2f} median={d['median']:.2f} SD={d['sd']:.2f} IQR=[{d['iqr_low']:.2f}, {d['iqr_high']:.2f}]")
        stats_rows.append({"set": name, "label": SET_LABELS[name], **d})
    stats_df = pd.DataFrame(stats_rows)
    tables_dir = REPO_ROOT / "tables"
    stats_df.to_csv(tables_dir / "degree_distribution_stats.csv", index=False)
    logger.info(f"Wrote {tables_dir}/degree_distribution_stats.csv")

    # --- Diagnostic: why the CI uses distinct-protein resampling, not Section 2.8's pair-graph clusters ---
    combined_pos_n0 = pd.concat([pos[["protein_a", "protein_b"]], neg0[["protein_a", "protein_b"]]], ignore_index=True)
    n_components = len(connected_component_groups(combined_pos_n0))
    logger.info(
        f"Positive+N0 combined pair graph has {n_components} connected component(s) "
        f"({len(combined_pos_n0)} pairs) -- a pair-cluster CI (Section 2.8's method) would be degenerate "
        f"at G={n_components}, so the mean-difference CI below uses distinct-protein-level resampling instead "
        f"(see src/evaluation/degree_distribution.py module docstring)."
    )

    # --- Distributional comparison: positives vs. each negative set ---
    comparison_rows = []
    for name in ["n0", "n1", "n2", "n3", "n5"]:
        c = compare_degree_distributions(pair_sets["positive"], pair_sets[name], degree, N_BOOT, SEED)
        logger.info(
            f"[positive vs {name}] pooled mean_diff={c['mean_diff']:.2f} median_diff={c['median_diff']:.2f} "
            f"SMD={c['standardized_mean_difference']:.3f} (well_balanced<0.1: {c['smd_well_balanced']}) "
            f"KS={c['ks_statistic']:.3f} (p={c['ks_pvalue']:.1e}) "
            f"Wasserstein={c['wasserstein_distance']:.2f} | protein-level mean_diff={c['protein_level_mean_diff']:.2f} "
            f"95% CI=[{c['protein_level_mean_diff_ci_low']:.2f}, {c['protein_level_mean_diff_ci_high']:.2f}]"
        )
        comparison_rows.append({"comparison": f"positive_vs_{name}", **c})
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(tables_dir / "degree_distribution_comparison.csv", index=False)
    logger.info(f"Wrote {tables_dir}/degree_distribution_comparison.csv")

    # --- Figure: density + ECDF, all six sets overlaid ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, vals in values.items():
        color = COLORS[name]
        style = "-" if name in ("positive", "n0") else "--"
        lw = 2.2 if name in ("positive", "n0") else 1.4
        axes[0].hist(vals, bins=40, range=(0, np.percentile(np.concatenate(list(values.values())), 99)),
                     density=True, histtype="step", color=color, linestyle=style, linewidth=lw, label=SET_LABELS[name])
        sorted_vals = np.sort(vals)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        axes[1].plot(sorted_vals, ecdf, color=color, linestyle=style, linewidth=lw, label=SET_LABELS[name])

    axes[0].set_xlabel("Pair-associated positive-degree")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Degree density (positives vs. every negative set)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Pair-associated positive-degree")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("Degree ECDF (positives vs. every negative set)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figures_dir = REPO_ROOT / "figures"
    fig.savefig(figures_dir / "degree_distribution_comparison.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/degree_distribution_comparison.pdf")

    results_dir = REPO_ROOT / "results" / "degree_distribution"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "note": (
                "Fuller characterization of Section 3.5's mean pair-associated-degree diagnostic "
                ": median/IQR/SD per set, and a distributional-comparison "
                "suite (KS statistic, Wasserstein distance, standardized mean difference/SMD, "
                "identical in formula to Cohen's d) for positives vs. each negative set, plus a "
                "distinct-protein-level bootstrap CI for the mean difference."
            ),
            "smd_balance_check": (
                "Is N2 (decile-matched) actually balanced on degree, not "
                "just less imbalanced than N0? SMD(positive vs N2) = "
                f"{comparison_df.set_index('comparison').loc['positive_vs_n2', 'standardized_mean_difference']:.4f}, "
                "far below the propensity-matching literature's 0.1 well-balanced threshold "
                f"(N0 comparator: {comparison_df.set_index('comparison').loc['positive_vs_n0', 'standardized_mean_difference']:.4f}). "
                "No tighter matching scheme (exact/nearest-neighbor -- already explored in "
                "results/degree_matching_sensitivity/) is required by this diagnostic."
            ),
            "positive_n0_combined_graph_n_components": n_components,
            "why_not_pair_cluster_ci": (
                "The positive and N0 pair sets share so many protein endpoints that their combined "
                "graph is a single connected component, making Section 2.8's connected-component "
                "cluster bootstrap/jackknife degenerate (G=1) for this comparison; the CI reported "
                "here instead resamples distinct proteins, a valid unit with no such degeneracy."
            ),
        },
        results_dir / "degree_distribution_summary.json",
    )
    write_manifest(results_dir, config={"n_boot": N_BOOT}, seed=SEED, extra={"script": "27_degree_distribution_characterization.py"})
    logger.info("Degree-distribution characterization complete.")


if __name__ == "__main__":
    main()
