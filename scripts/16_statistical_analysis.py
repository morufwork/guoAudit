"""Statistical evaluation.

Applies two confidence-interval methods that respect PPI-pair dependence
(both built on connected-component clustering, see
src/evaluation/bootstrap.py) to the project's central open questions:

  Q1: Is each architecture's R3 performance under N2 actually
      distinguishable from chance (ROC-AUC=0.5, MCC=0, PR-AUC=test-set
      prevalence), or does the point estimate's apparent signal disappear
      once uncertainty is accounted for? PR-AUC is added alongside ROC-AUC/MCC because negative-sampling
      construction directly changes the class-conditional score
      distribution -- exactly the kind of shift PR-AUC is sensitive to
      and ROC-AUC is comparatively insensitive to -- even though this
      benchmark's own class balance stays close to 50:50 throughout
      (N0 test prevalence 0.483, N2 exactly 0.500), so PR-AUC's chance
      baseline (the test set's own positive prevalence, not a fixed 0.5)
      is used as its null.
  Q2: Is the architecture improvement (Siamese MLP vs. Logistic Regression)
      statistically significant, separately under N0 and N2? (Decision
      Gate 2, now with a confidence interval instead of a single point
      estimate each side of the N0/N2 comparison.)
  Q3: Is the N0 -> N2 collapse itself outside plausible sampling noise, or
      could point estimates this different arise by chance given the
      dependent pair structure? (N0 and N2 have different, non-overlapping
      test sets, so this is an unpaired/independent-samples comparison, not
      a paired one -- variance combined from each side.)

Both a percentile cluster bootstrap (2,000 replicates, resampling whole
connected components with replacement) and a delete-1-cluster jackknife
(t(G_eff-1) reference distribution) are computed and reported side by side.
They are expected to diverge: this benchmark's R3 test sets are each
dominated by one giant connected component with only a handful of other
components alongside it (5-19 usable components total per test set), the
"few clusters" regime in which a percentile cluster bootstrap is known to
be erratic, while the jackknife is the standard small-G remedy but produces
much wider intervals as an honest consequence of there being so few
independent units to resample. See src/evaluation/bootstrap.py's module
docstring for the full history (including an earlier, buggy protein_a-only
grouping this replaced).

Multiple comparisons: Q1+Q2+Q3 together run
44 hypothesis tests per CI method (12 Q1 ROC-AUC + 12 Q1 MCC + 12 Q1 PR-AUC
+ 2 Q2 + 6 Q3), each against its own chance/no-effect null. A per-test 95%
CI alone does not control the family-wise false-discovery rate across that
many comparisons, so a Benjamini-Hochberg FDR correction (alpha=0.05) is
applied within each CI method's full battery of 44 tests (one family per
method -- bootstrap p-values and jackknife p-values are not pooled into one
family, since the two methods have different small-sample behavior at this
benchmark's cluster counts). Both the uncorrected per-test call and the
BH-corrected call are reported; the manuscript's headline claims use the
corrected one. See tables/statistical_test_family.csv
(scripts/29_statistical_test_family_table.py) for the full pre-specified
family definition.

Writes:
  results/statistical_analysis.json
  tables/table10_statistical_comparison.csv
  figures/figure_bootstrap_ci.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from src.evaluation.bootstrap import (
    benjamini_hochberg,
    bootstrap_ci,
    cluster_jackknife_ci,
    connected_component_groups,
    paired_bootstrap_diff_ci,
    paired_cluster_jackknife_diff_ci,
    protein_cluster_bootstrap_indices,
    unpaired_cluster_jackknife_diff_ci,
)
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("statistical_analysis")

N_BOOT = 2000
SEED = 42
BH_ALPHA = 0.05
MODEL_ORDER = ["logistic_regression", "random_forest", "mlp", "siamese_mlp", "attention_fusion", "cross_interaction"]


def mcc_metric(y_true, y_prob, threshold=0.5):
    return matthews_corrcoef(y_true, (y_prob >= threshold).astype(int))


def load_predictions(neg_name: str, model_name: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "results" / "architecture_ablation" / f"{neg_name}__{model_name}" / "predictions.csv")


def empirical_two_sided_p(values: np.ndarray, null: float) -> float:
    """Same construction as bootstrap.py's internal helper, needed here
    too because Q3's bootstrap diff distribution is built by hand (paired
    per-replicate subtraction across two independent index sets) rather
    than through paired_bootstrap_diff_ci."""
    p_le = np.mean(values <= null)
    p_ge = np.mean(values >= null)
    return float(min(1.0, 2 * min(p_le, p_ge)))


def apply_bh_family(records: list[dict], alpha: float = BH_ALPHA) -> None:
    """In place: adds 'p_adjusted' and 'significant_bh' to each result dict
    in `records`, treating them as one family of hypothesis tests under
    Benjamini-Hochberg FDR control."""
    p_values = [r["p_value"] for r in records]
    reject, adjusted = benjamini_hochberg(p_values, alpha=alpha)
    for r, rej, adj in zip(records, reject, adjusted):
        r["p_adjusted"] = adj
        r["significant_bh"] = rej


def main():
    bootstrap_family, jackknife_family = [], []  # every test's result dict, for BH correction across all of Q1+Q2+Q3

    # --- Q1: per-model CI under each negative set ---
    q1 = {}
    boot_indices_cache, groups_cache = {}, {}
    for neg_name in ["n0", "n2"]:
        for model_name in MODEL_ORDER:
            pred = load_predictions(neg_name, model_name).reset_index(drop=True)
            y_true, y_prob = pred["y_true"].to_numpy(), pred["y_probability"].to_numpy()

            cache_key = (neg_name, model_name)
            idx = protein_cluster_bootstrap_indices(pred, N_BOOT, SEED)
            groups = connected_component_groups(pred)
            boot_indices_cache[cache_key] = idx
            groups_cache[cache_key] = groups

            prevalence = float(y_true.mean())  # PR-AUC's chance baseline is the test set's own positive rate, not a fixed 0.5
            roc_boot = bootstrap_ci(y_true, y_prob, idx, roc_auc_score, null=0.5)
            mcc_boot = bootstrap_ci(y_true, y_prob, idx, mcc_metric, null=0.0)
            pr_boot = bootstrap_ci(y_true, y_prob, idx, average_precision_score, null=prevalence)
            roc_jk = cluster_jackknife_ci(y_true, y_prob, groups, roc_auc_score, null=0.5)
            mcc_jk = cluster_jackknife_ci(y_true, y_prob, groups, mcc_metric, null=0.0)
            pr_jk = cluster_jackknife_ci(y_true, y_prob, groups, average_precision_score, null=prevalence)
            q1[cache_key] = {
                "roc_boot": roc_boot, "mcc_boot": mcc_boot, "pr_boot": pr_boot,
                "roc_jk": roc_jk, "mcc_jk": mcc_jk, "pr_jk": pr_jk, "prevalence": prevalence,
            }
            bootstrap_family += [roc_boot, mcc_boot, pr_boot]
            jackknife_family += [roc_jk, mcc_jk, pr_jk]

    # --- Q2: paired comparison, architecture (siamese_mlp) vs. baseline (logistic_regression), same test set ---
    q2 = {}
    for neg_name in ["n0", "n2"]:
        pred_lr = load_predictions(neg_name, "logistic_regression").reset_index(drop=True)
        pred_siamese = load_predictions(neg_name, "siamese_mlp").reset_index(drop=True)
        assert len(pred_lr) == len(pred_siamese) and (pred_lr["protein_a"] == pred_siamese["protein_a"]).all(), \
            "logistic_regression and siamese_mlp must share the same test set for a paired comparison"

        y_true = pred_lr["y_true"].to_numpy()
        idx = boot_indices_cache[(neg_name, "logistic_regression")]
        groups = groups_cache[(neg_name, "logistic_regression")]
        diff_boot = paired_bootstrap_diff_ci(
            y_true, pred_siamese["y_probability"].to_numpy(), pred_lr["y_probability"].to_numpy(), idx, roc_auc_score,
        )
        diff_jk = paired_cluster_jackknife_diff_ci(
            y_true, pred_siamese["y_probability"].to_numpy(), pred_lr["y_probability"].to_numpy(), groups, roc_auc_score,
        )
        q2[neg_name] = {"boot": diff_boot, "jk": diff_jk}
        bootstrap_family.append(diff_boot)
        jackknife_family.append(diff_jk)

    # --- Q3: N0 vs N2 collapse, independent-samples comparison (different test sets) ---
    q3 = {}
    for model_name in MODEL_ORDER:
        pred_n0 = load_predictions("n0", model_name).reset_index(drop=True)
        pred_n2 = load_predictions("n2", model_name).reset_index(drop=True)
        idx_n0, idx_n2 = boot_indices_cache[("n0", model_name)], boot_indices_cache[("n2", model_name)]
        groups_n0, groups_n2 = groups_cache[("n0", model_name)], groups_cache[("n2", model_name)]

        y_true_n0, y_prob_n0 = pred_n0["y_true"].to_numpy(), pred_n0["y_probability"].to_numpy()
        y_true_n2, y_prob_n2 = pred_n2["y_true"].to_numpy(), pred_n2["y_probability"].to_numpy()

        boot_n0 = np.array([roc_auc_score(y_true_n0[i], y_prob_n0[i]) for i in idx_n0 if len(np.unique(y_true_n0[i])) > 1])
        boot_n2 = np.array([roc_auc_score(y_true_n2[i], y_prob_n2[i]) for i in idx_n2 if len(np.unique(y_true_n2[i])) > 1])
        m = min(len(boot_n0), len(boot_n2))
        diff_dist = boot_n0[:m] - boot_n2[:m]
        lower, upper = np.percentile(diff_dist, [2.5, 97.5])
        point_diff = roc_auc_score(y_true_n0, y_prob_n0) - roc_auc_score(y_true_n2, y_prob_n2)
        diff_boot = {
            "point": float(point_diff), "ci_low": float(lower), "ci_high": float(upper),
            "significant": bool(lower > 0 or upper < 0),
            "p_value": empirical_two_sided_p(diff_dist, 0.0),
        }
        diff_jk = unpaired_cluster_jackknife_diff_ci(
            y_true_n0, y_prob_n0, groups_n0, y_true_n2, y_prob_n2, groups_n2, roc_auc_score,
        )
        q3[model_name] = {"boot": diff_boot, "jk": diff_jk}
        bootstrap_family.append(diff_boot)
        jackknife_family.append(diff_jk)

    # --- Multiple-comparisons correction: BH-FDR within each CI method's full battery ---
    apply_bh_family(bootstrap_family)
    apply_bh_family(jackknife_family)
    logger.info(
        f"BH-FDR correction (alpha={BH_ALPHA}) applied within {len(bootstrap_family)} bootstrap tests and "
        f"{len(jackknife_family)} jackknife tests (Q1 ROC-AUC + Q1 MCC + Q1 PR-AUC + Q2 + Q3, each CI method its own family)."
    )

    # --- Assemble results/tables_rows now that every dict carries p_value + BH fields ---
    results = {"per_model_ci": [], "architecture_vs_lr_paired": [], "n0_vs_n2_unpaired": []}
    tables_rows = []

    for neg_name in ["n0", "n2"]:
        for model_name in MODEL_ORDER:
            r = q1[(neg_name, model_name)]
            roc_boot, mcc_boot, pr_boot = r["roc_boot"], r["mcc_boot"], r["pr_boot"]
            roc_jk, mcc_jk, pr_jk = r["roc_jk"], r["mcc_jk"], r["pr_jk"]
            logger.info(
                f"[{neg_name} x {model_name}] ROC-AUC={roc_boot['point']:.3f} "
                f"bootstrap=[{roc_boot['ci_low']:.3f}, {roc_boot['ci_high']:.3f}]"
                f"{'*' if roc_boot['significant_bh'] else ''} "
                f"jackknife=[{roc_jk['ci_low']:.3f}, {roc_jk['ci_high']:.3f}]"
                f"{'*' if roc_jk['significant_bh'] else ''} "
                f"| PR-AUC={pr_boot['point']:.3f} (chance={r['prevalence']:.3f}) "
                f"bootstrap=[{pr_boot['ci_low']:.3f}, {pr_boot['ci_high']:.3f}]"
                f"{'*' if pr_boot['significant_bh'] else ''} "
                f"jackknife=[{pr_jk['ci_low']:.3f}, {pr_jk['ci_high']:.3f}]"
                f"{'*' if pr_jk['significant_bh'] else ''} "
                f"({roc_jk['n_clusters_used']}/{roc_jk['n_clusters']} components) [BH-corrected]"
            )
            results["per_model_ci"].append({
                "negative_set": neg_name, "model": model_name, "prevalence": r["prevalence"],
                "roc_auc_bootstrap": roc_boot, "mcc_bootstrap": mcc_boot, "pr_auc_bootstrap": pr_boot,
                "roc_auc_jackknife": roc_jk, "mcc_jackknife": mcc_jk, "pr_auc_jackknife": pr_jk,
                "roc_auc_significantly_above_chance_bootstrap": roc_boot["significant_bh"],
                "mcc_significantly_above_chance_bootstrap": mcc_boot["significant_bh"],
                "pr_auc_significantly_above_chance_bootstrap": pr_boot["significant_bh"],
                "roc_auc_significantly_above_chance_jackknife": roc_jk["significant_bh"],
                "mcc_significantly_above_chance_jackknife": mcc_jk["significant_bh"],
                "pr_auc_significantly_above_chance_jackknife": pr_jk["significant_bh"],
            })
            tables_rows.append({
                "negative_set": neg_name, "model": model_name, "prevalence": r["prevalence"],
                "roc_auc": roc_boot["point"],
                "roc_auc_bootstrap_ci_low": roc_boot["ci_low"], "roc_auc_bootstrap_ci_high": roc_boot["ci_high"],
                "roc_auc_bootstrap_p": roc_boot["p_value"], "roc_auc_bootstrap_p_adj": roc_boot["p_adjusted"],
                "roc_auc_bootstrap_sig_above_chance": roc_boot["significant_bh"],
                "roc_auc_jackknife_ci_low": roc_jk["ci_low"], "roc_auc_jackknife_ci_high": roc_jk["ci_high"],
                "roc_auc_jackknife_p": roc_jk["p_value"], "roc_auc_jackknife_p_adj": roc_jk["p_adjusted"],
                "roc_auc_jackknife_sig_above_chance": roc_jk["significant_bh"],
                "pr_auc": pr_boot["point"],
                "pr_auc_bootstrap_ci_low": pr_boot["ci_low"], "pr_auc_bootstrap_ci_high": pr_boot["ci_high"],
                "pr_auc_bootstrap_p": pr_boot["p_value"], "pr_auc_bootstrap_p_adj": pr_boot["p_adjusted"],
                "pr_auc_bootstrap_sig_above_chance": pr_boot["significant_bh"],
                "pr_auc_jackknife_ci_low": pr_jk["ci_low"], "pr_auc_jackknife_ci_high": pr_jk["ci_high"],
                "pr_auc_jackknife_p": pr_jk["p_value"], "pr_auc_jackknife_p_adj": pr_jk["p_adjusted"],
                "pr_auc_jackknife_sig_above_chance": pr_jk["significant_bh"],
                "mcc": mcc_boot["point"],
                "mcc_bootstrap_ci_low": mcc_boot["ci_low"], "mcc_bootstrap_ci_high": mcc_boot["ci_high"],
                "mcc_bootstrap_p": mcc_boot["p_value"], "mcc_bootstrap_p_adj": mcc_boot["p_adjusted"],
                "mcc_bootstrap_sig_above_chance": mcc_boot["significant_bh"],
                "mcc_jackknife_ci_low": mcc_jk["ci_low"], "mcc_jackknife_ci_high": mcc_jk["ci_high"],
                "mcc_jackknife_p": mcc_jk["p_value"], "mcc_jackknife_p_adj": mcc_jk["p_adjusted"],
                "mcc_jackknife_sig_above_chance": mcc_jk["significant_bh"],
                "n_clusters": roc_jk["n_clusters"], "n_clusters_used": roc_jk["n_clusters_used"],
            })

    for neg_name in ["n0", "n2"]:
        diff_boot, diff_jk = q2[neg_name]["boot"], q2[neg_name]["jk"]
        logger.info(
            f"[{neg_name}] Siamese MLP - Logistic Regression ROC-AUC delta = {diff_boot['point']:.3f} "
            f"bootstrap=[{diff_boot['ci_low']:.3f}, {diff_boot['ci_high']:.3f}]"
            f"{'*' if diff_boot['significant_bh'] else ''} "
            f"jackknife=[{diff_jk['ci_low']:.3f}, {diff_jk['ci_high']:.3f}]"
            f"{'*' if diff_jk['significant_bh'] else ''} [BH-corrected]"
        )
        results["architecture_vs_lr_paired"].append({
            "negative_set": neg_name,
            "delta_roc_auc_bootstrap": {**diff_boot, "significant": diff_boot["significant_bh"]},
            "delta_roc_auc_jackknife": {**diff_jk, "significant": diff_jk["significant_bh"]},
        })

    for model_name in MODEL_ORDER:
        diff_boot, diff_jk = q3[model_name]["boot"], q3[model_name]["jk"]
        logger.info(
            f"[{model_name}] N0 - N2 ROC-AUC delta (unpaired) = {diff_boot['point']:.3f} "
            f"bootstrap=[{diff_boot['ci_low']:.3f}, {diff_boot['ci_high']:.3f}]{'*' if diff_boot['significant_bh'] else ''} "
            f"jackknife=[{diff_jk['ci_low']:.3f}, {diff_jk['ci_high']:.3f}]{'*' if diff_jk['significant_bh'] else ''} [BH-corrected]"
        )
        results["n0_vs_n2_unpaired"].append({
            "model": model_name,
            "delta_roc_auc_bootstrap": {**diff_boot, "significant": diff_boot["significant_bh"]},
            "delta_roc_auc_jackknife": {**diff_jk, "significant": diff_jk["significant_bh"]},
        })

    results["multiple_comparisons"] = {
        "method": "Benjamini-Hochberg FDR",
        "alpha": BH_ALPHA,
        "n_tests_per_family": len(bootstrap_family),
        "families": ["bootstrap (Q1 ROC-AUC + Q1 MCC + Q1 PR-AUC + Q2 + Q3)", "jackknife (Q1 ROC-AUC + Q1 MCC + Q1 PR-AUC + Q2 + Q3)"],
        "note": "significant_bh fields above are BH-corrected within their CI method's family; "
                "p_value fields are the uncorrected per-test two-sided p-value.",
    }

    save_json(results, REPO_ROOT / "results" / "statistical_analysis.json")
    tables_dir = REPO_ROOT / "tables"
    pd.DataFrame(tables_rows).to_csv(tables_dir / "table10_statistical_comparison.csv", index=False)
    logger.info(f"Wrote results/statistical_analysis.json and {tables_dir}/table10_statistical_comparison.csv")

    # --- Figure: ROC-AUC with 95% CI (jackknife), N0 vs N2, per model ---
    figures_dir = REPO_ROOT / "figures"
    fig, ax = plt.subplots(figsize=(10, 5))
    df = pd.DataFrame(tables_rows)
    x = np.arange(len(MODEL_ORDER))
    for i, neg_name in enumerate(["n0", "n2"]):
        sub = df[df["negative_set"] == neg_name].set_index("model").loc[MODEL_ORDER]
        offset = (i - 0.5) * 0.35
        yerr = np.vstack([sub["roc_auc"] - sub["roc_auc_jackknife_ci_low"], sub["roc_auc_jackknife_ci_high"] - sub["roc_auc"]])
        ax.errorbar(x + offset, sub["roc_auc"], yerr=yerr, fmt="o", capsize=4, label=neg_name.upper())
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
    ax.set_ylabel("ROC-AUC [95% CI, cluster jackknife]")
    ax.set_title("95% CIs, R3, connected-component cluster jackknife")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_bootstrap_ci.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/figure_bootstrap_ci.pdf")

    write_manifest(
        REPO_ROOT / "results",
        config={
            "n_boot": N_BOOT,
            "resampling": "connected_component (transitive protein clusters)",
            "ci_methods": ["percentile cluster bootstrap", "delete-1-cluster jackknife, t(G_eff-1)"],
            "multiple_comparisons": f"Benjamini-Hochberg FDR, alpha={BH_ALPHA}, per-CI-method family",
        },
        seed=SEED,
        extra={"script": "16_statistical_analysis.py"},
    )
    logger.info("Statistical analysis complete.")


if __name__ == "__main__":
    main()
