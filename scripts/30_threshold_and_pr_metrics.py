"""Threshold justification and operating-point metrics.

Item 1.11 asks why the classification threshold is fixed at 0.5, given
that a poorly-calibrated model's raw 0.5 cutoff may not correspond to a
meaningful decision boundary, and recommends reporting MCC at a threshold
selected on validation data (not the test set) as a complementary check.
Item 1.10 asks for operating-point metrics beyond ROC-AUC/MCC given this
benchmark's artificial 50:50 class balance -- precision at fixed recall
and recall at fixed precision specifically.

Rather than a fresh threshold-selection experiment (which would need its
own held-out split and introduce new researcher degrees of freedom -- grid
resolution, tie-breaking rule), this script reuses the calibration analysis's ALREADY-FIT
Platt/isotonic calibrators (results/calibration/<negset>__<model>/predictions.csv),
which exist specifically to make a fixed threshold meaningful: a calibrator
fit on a held-out slice of training data (never the test set) remaps raw scores so 0.5 corresponds to the correct
posterior probability, if calibration worked. MCC at 0.5 on the raw,
Platt-calibrated, and isotonic-calibrated probabilities are reported side
by side: if MCC-at-0.5 changes materially between raw and calibrated
probabilities, that IS evidence the raw threshold's meaningfulness was in
question; if it does not, 0.5 was already a reasonable operating point for
that model.

Precision-at-recall and recall-at-precision are computed directly from
the architecture ablation's already-saved test predictions (no new experiment, no
threshold-tuning-on-test-data concern: the target recall/precision level
is fixed in advance, not selected to maximize anything on this test set).

Writes:
  tables/threshold_and_operating_point_metrics.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_recall_curve

from src.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("threshold_and_pr_metrics")

MODEL_ORDER = ["logistic_regression", "random_forest", "mlp", "siamese_mlp", "attention_fusion", "cross_interaction"]
TARGET_RECALL = 0.5
TARGET_PRECISION = 0.5


def mcc_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    return matthews_corrcoef(y_true, (y_prob >= threshold).astype(int))


def precision_at_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float) -> float:
    """Precision at the operating point where recall first reaches (>=)
    target_recall, scanning thresholds from most to least permissive
    (sklearn's precision_recall_curve returns recall in decreasing order)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    idx = np.where(recall >= target_recall)[0]
    if len(idx) == 0:
        return float("nan")  # target recall unreachable at any threshold
    return float(precision[idx[-1]])  # last index with recall >= target = tightest threshold still meeting it


def recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    idx = np.where(precision >= target_precision)[0]
    if len(idx) == 0:
        return float("nan")
    return float(recall[idx[0]])  # first index (highest recall) meeting the precision floor


def main():
    rows = []
    for neg_name in ["n0", "n2"]:
        for model_name in MODEL_ORDER:
            p10 = pd.read_csv(REPO_ROOT / "results" / "architecture_ablation" / f"{neg_name}__{model_name}" / "predictions.csv")
            p11 = pd.read_csv(REPO_ROOT / "results" / "calibration" / f"{neg_name}__{model_name}" / "predictions.csv")

            y_true_10, y_prob_10 = p10["y_true"].to_numpy(), p10["y_probability"].to_numpy()
            y_true_11 = p11["y_true"].to_numpy()

            # Calibration effects must compare scores from the same fitted model.
            # the architecture ablation remains the source only for the descriptive PR operating points.
            mcc_raw = mcc_at(y_true_11, p11["y_probability_raw"].to_numpy())
            mcc_platt = mcc_at(y_true_11, p11["y_probability_platt"].to_numpy())
            mcc_isotonic = mcc_at(y_true_11, p11["y_probability_isotonic"].to_numpy())
            prec_at_recall = precision_at_recall(y_true_10, y_prob_10, TARGET_RECALL)
            rec_at_precision = recall_at_precision(y_true_10, y_prob_10, TARGET_PRECISION)

            logger.info(
                f"[{neg_name} x {model_name}] MCC@0.5: raw={mcc_raw:.3f} platt={mcc_platt:.3f} isotonic={mcc_isotonic:.3f} | "
                f"precision@recall={TARGET_RECALL}={prec_at_recall:.3f} recall@precision={TARGET_PRECISION}={rec_at_precision:.3f}"
            )
            rows.append({
                "negative_set": neg_name, "model": model_name,
                "mcc_at_0.5_raw": mcc_raw, "mcc_at_0.5_platt": mcc_platt, "mcc_at_0.5_isotonic": mcc_isotonic,
                "mcc_raw_vs_calibrated_max_abs_diff": max(abs(mcc_raw - mcc_platt), abs(mcc_raw - mcc_isotonic)),
                f"precision_at_recall_{TARGET_RECALL}": prec_at_recall,
                f"recall_at_precision_{TARGET_PRECISION}": rec_at_precision,
            })

    df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    df.to_csv(tables_dir / "threshold_and_operating_point_metrics.csv", index=False)
    logger.info(f"Wrote {tables_dir}/threshold_and_operating_point_metrics.csv")


if __name__ == "__main__":
    main()
