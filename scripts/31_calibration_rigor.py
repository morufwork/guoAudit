"""Calibration rigor: bin-strategy sensitivity, cluster-aware CIs, and a
separation check for the Cox regression.

Problem 1 (ECE is bin-dependent): reports ECE at n_bins in {5, 10, 15, 20}
under both equal-width (the headline Table 8 convention) and equal-frequency
binning, so the reported "ECE=10 bins, equal-width" number's sensitivity to
that choice is visible rather than asserted.

Problem 2 (calibration on a small R3 test set): every calibration metric
this project reports (ECE, Brier, Cox slope/intercept) has so far been a
bare point estimate, unlike every other headline number in this study
(Section 3.8 onward), which carries a connected-component cluster-aware CI.
This script closes that gap, reusing the exact same tested
`bootstrap_ci`/`cluster_jackknife_ci` machinery from src/evaluation/bootstrap.py
on the calibration analysis's already-saved raw test predictions -- no retraining needed.

Problem 4 (Cox regression rigor): checks `calibration_slope_intercept`'s
`converged`/`possible_separation` diagnostic on all 12 real (negative_set,
model) combinations and reports the result directly, rather than assuming
the unpenalized fit behaved well.

Problem 3 (N2's near-zero ECE is a base-rate artifact, not good calibration)
is a framing fix made directly in the manuscript text (Section 3.10), not a
new computation -- restated here in the module docstring as the reason the
tables below report N0 and N2 side by side rather than N2 in isolation.

Writes:
  tables/ece_bin_sensitivity.csv
  tables/calibration_metrics_with_ci.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import brier_score_loss

from src.evaluation.bootstrap import bootstrap_ci, cluster_jackknife_ci, connected_component_groups, protein_cluster_bootstrap_indices
from src.evaluation.calibration import (
    calibration_slope_intercept,
    expected_calibration_error,
    expected_calibration_error_equal_frequency,
)
from src.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("calibration_rigor")

N_BOOT = 2000
SEED = 42
BIN_COUNTS = [5, 10, 15, 20]
MODEL_ORDER = ["logistic_regression", "random_forest", "mlp", "siamese_mlp", "attention_fusion", "cross_interaction"]


def ece_10bin(y_true, y_prob):
    return expected_calibration_error(y_true, y_prob, n_bins=10)


def slope_metric(y_true, y_prob):
    return calibration_slope_intercept(y_true, y_prob)["calibration_slope"]


def intercept_metric(y_true, y_prob):
    return calibration_slope_intercept(y_true, y_prob)["calibration_intercept"]


def main():
    sensitivity_rows, ci_rows, separation_rows = [], [], []

    for neg_name in ["n0", "n2"]:
        for model_name in MODEL_ORDER:
            pred = pd.read_csv(REPO_ROOT / "results" / "calibration" / f"{neg_name}__{model_name}" / "predictions.csv").reset_index(drop=True)
            y_true, y_prob = pred["y_true"].to_numpy(), pred["y_probability_raw"].to_numpy()

            # --- Problem 1: ECE bin-count / strategy sensitivity ---
            for n_bins in BIN_COUNTS:
                ece_w = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
                ece_f = expected_calibration_error_equal_frequency(y_true, y_prob, n_bins=n_bins)
                sensitivity_rows.append({
                    "negative_set": neg_name, "model": model_name, "n_bins": n_bins,
                    "ece_equal_width": ece_w, "ece_equal_frequency": ece_f,
                })

            # --- Problem 2: cluster-aware CIs for ECE(10-bin, equal-width), Brier, Cox slope/intercept ---
            idx = protein_cluster_bootstrap_indices(pred, N_BOOT, SEED)
            groups = connected_component_groups(pred)

            ece_boot = bootstrap_ci(y_true, y_prob, idx, ece_10bin, null=0.0)
            ece_jk = cluster_jackknife_ci(y_true, y_prob, groups, ece_10bin, null=0.0)
            brier_boot = bootstrap_ci(y_true, y_prob, idx, brier_score_loss, null=0.25)
            brier_jk = cluster_jackknife_ci(y_true, y_prob, groups, brier_score_loss, null=0.25)
            slope_boot = bootstrap_ci(y_true, y_prob, idx, slope_metric, null=1.0)
            slope_jk = cluster_jackknife_ci(y_true, y_prob, groups, slope_metric, null=1.0)
            intercept_boot = bootstrap_ci(y_true, y_prob, idx, intercept_metric, null=0.0)
            intercept_jk = cluster_jackknife_ci(y_true, y_prob, groups, intercept_metric, null=0.0)

            logger.info(
                f"[{neg_name} x {model_name}] ECE(10-bin)={ece_boot['point']:.3f} "
                f"bootstrap=[{ece_boot['ci_low']:.3f}, {ece_boot['ci_high']:.3f}] "
                f"jackknife=[{ece_jk['ci_low']:.3f}, {ece_jk['ci_high']:.3f}] | "
                f"Brier={brier_boot['point']:.3f} bootstrap=[{brier_boot['ci_low']:.3f}, {brier_boot['ci_high']:.3f}] | "
                f"slope={slope_boot['point']:.3f} bootstrap=[{slope_boot['ci_low']:.3f}, {slope_boot['ci_high']:.3f}] "
                f"jackknife=[{slope_jk['ci_low']:.3f}, {slope_jk['ci_high']:.3f}]"
            )
            ci_rows.append({
                "negative_set": neg_name, "model": model_name,
                "ece_point": ece_boot["point"],
                "ece_bootstrap_ci_low": ece_boot["ci_low"], "ece_bootstrap_ci_high": ece_boot["ci_high"],
                "ece_jackknife_ci_low": ece_jk["ci_low"], "ece_jackknife_ci_high": ece_jk["ci_high"],
                "brier_point": brier_boot["point"],
                "brier_bootstrap_ci_low": brier_boot["ci_low"], "brier_bootstrap_ci_high": brier_boot["ci_high"],
                "brier_jackknife_ci_low": brier_jk["ci_low"], "brier_jackknife_ci_high": brier_jk["ci_high"],
                "cox_slope_point": slope_boot["point"],
                "cox_slope_bootstrap_ci_low": slope_boot["ci_low"], "cox_slope_bootstrap_ci_high": slope_boot["ci_high"],
                "cox_slope_jackknife_ci_low": slope_jk["ci_low"], "cox_slope_jackknife_ci_high": slope_jk["ci_high"],
                "cox_intercept_point": intercept_boot["point"],
                "cox_intercept_bootstrap_ci_low": intercept_boot["ci_low"], "cox_intercept_bootstrap_ci_high": intercept_boot["ci_high"],
                "cox_intercept_jackknife_ci_low": intercept_jk["ci_low"], "cox_intercept_jackknife_ci_high": intercept_jk["ci_high"],
                "n_clusters": ece_jk["n_clusters"], "n_clusters_used": ece_jk["n_clusters_used"],
            })

            # --- Problem 4: separation check on the real fit ---
            diag = calibration_slope_intercept(y_true, y_prob)
            separation_rows.append({
                "negative_set": neg_name, "model": model_name,
                "converged": diag["converged"], "possible_separation": diag["possible_separation"],
                "slope": diag["calibration_slope"], "intercept": diag["calibration_intercept"],
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    ci_df = pd.DataFrame(ci_rows)
    separation_df = pd.DataFrame(separation_rows)

    tables_dir = REPO_ROOT / "tables"
    sensitivity_df.to_csv(tables_dir / "ece_bin_sensitivity.csv", index=False)
    ci_df.to_csv(tables_dir / "calibration_metrics_with_ci.csv", index=False)
    logger.info(f"Wrote {tables_dir}/ece_bin_sensitivity.csv and {tables_dir}/calibration_metrics_with_ci.csv")

    n_separation = int(separation_df["possible_separation"].sum())
    n_nonconverged = int((~separation_df["converged"]).sum())
    logger.info(
        f"Cox regression separation check across all {len(separation_df)} (negative_set, model) fits: "
        f"{n_separation} flagged possible_separation, {n_nonconverged} failed to converge."
    )
    if n_separation or n_nonconverged:
        logger.info(separation_df[separation_df["possible_separation"] | ~separation_df["converged"]].to_string())


if __name__ == "__main__":
    main()
