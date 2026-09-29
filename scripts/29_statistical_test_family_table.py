"""Statistical-analysis table: pre-specified hypothesis family, one row per
test.

The manuscript's Benjamini-Hochberg correction (Section 2.8) needs
an explicit, pre-specified definition of what counts as one "family" of
hypothesis tests -- otherwise a reader has no way to tell the family
boundary from a choice made after seeing results. This script does not
compute anything new: it re-documents the family structure already fixed
in scripts/16_statistical_analysis.py (results/statistical_analysis.json)
as an explicit, auditable table -- one row per individual test, with its
question, null hypothesis, metric, comparison, method, and which family it
was corrected within.

The family boundaries reproduced here were fixed in advance (six
pre-specified architectures, PR-AUC/ROC-AUC/MCC as primary/secondary
metrics, correction for multiple comparisons), not chosen
after inspecting Section 3.8's results -- see the manuscript's Section 2.8
for the prose justification this table supports. PR-AUC joined the family alongside ROC-AUC/MCC, adding 12 more Q1
tests per method (44 total per method, 88 across both).

Writes:
  tables/statistical_test_family.csv   (88 rows: every individual test)
  tables/statistical_test_family_summary.csv  (4 rows: one per question type)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("statistical_test_family_table")

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "mlp": "MLP",
    "siamese_mlp": "Siamese MLP", "attention_fusion": "Attention Fusion", "cross_interaction": "Cross-Interaction",
}


def main():
    with open(REPO_ROOT / "results" / "statistical_analysis.json") as f:
        results = json.load(f)
    rows = []

    # --- Q1: per-model chance tests (12 model x negset combos x 3 metrics = 36 tests) ---
    for item in results["per_model_ci"]:
        negset, model = item["negative_set"].upper(), MODEL_LABELS[item["model"]]
        prevalence = item.get("prevalence")
        metrics = [
            ("ROC-AUC", "AUC = 0.5", "roc_auc"),
            ("MCC", "MCC = 0", "mcc"),
            ("PR-AUC", f"PR-AUC = test-set prevalence ({prevalence:.3f})" if prevalence is not None else "PR-AUC = test-set prevalence", "pr_auc"),
        ]
        for metric, null_desc, key_prefix in metrics:
            for method in ["bootstrap", "jackknife"]:
                r = item[f"{key_prefix}_{method}"]
                rows.append({
                    "test_id": f"Q1-{key_prefix}-{item['negative_set']}-{item['model']}-{method}",
                    "question": "Q1: distinguishable from chance?",
                    "null_hypothesis": f"{model} under {negset}: {null_desc}",
                    "metric": metric, "negative_set": negset, "comparison": f"{model} vs. chance",
                    "method": method, "family": f"{method} (44-test)",
                    "point": r.get("point"), "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                    "p_value_raw": r.get("p_value"), "p_value_adjusted": r.get("p_adjusted"),
                    "significant_after_bh": r.get("significant_bh"),
                })

    # --- Q2: paired architecture-vs-baseline (2 negsets x 2 methods = 4 tests) ---
    for item in results["architecture_vs_lr_paired"]:
        negset = item["negative_set"].upper()
        for method in ["bootstrap", "jackknife"]:
            r = item[f"delta_roc_auc_{method}"]
            rows.append({
                "test_id": f"Q2-{item['negative_set']}-{method}",
                "question": "Q2: architecture vs. baseline (paired, same test set)",
                "null_hypothesis": f"Siamese MLP - Logistic Regression ROC-AUC = 0, under {negset}",
                "metric": "ΔROC-AUC", "negative_set": negset, "comparison": "Siamese MLP vs. Logistic Regression",
                "method": method, "family": f"{method} (44-test)",
                "point": r.get("point"), "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                "p_value_raw": r.get("p_value"), "p_value_adjusted": r.get("p_adjusted"),
                "significant_after_bh": r.get("significant"),
            })

    # --- Q3: unpaired N0-vs-N2 (6 models x 2 methods = 12 tests) ---
    for item in results["n0_vs_n2_unpaired"]:
        model = MODEL_LABELS[item["model"]]
        for method in ["bootstrap", "jackknife"]:
            r = item[f"delta_roc_auc_{method}"]
            rows.append({
                "test_id": f"Q3-{item['model']}-{method}",
                "question": "Q3: N0 vs. N2 (unpaired, independent test sets)",
                "null_hypothesis": f"{model}: ROC-AUC(N0) - ROC-AUC(N2) = 0",
                "metric": "ΔROC-AUC", "negative_set": "N0 vs. N2", "comparison": model,
                "method": method, "family": f"{method} (44-test)",
                "point": r.get("point"), "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                "p_value_raw": r.get("p_value"), "p_value_adjusted": r.get("p_adjusted"),
                "significant_after_bh": r.get("significant"),
            })

    df = pd.DataFrame(rows)
    n_bootstrap = (df["method"] == "bootstrap").sum()
    n_jackknife = (df["method"] == "jackknife").sum()
    assert n_bootstrap == 44 and n_jackknife == 44, f"expected 44+44=88 tests, got {n_bootstrap}+{n_jackknife}"

    tables_dir = REPO_ROOT / "tables"
    df.to_csv(tables_dir / "statistical_test_family.csv", index=False)
    logger.info(f"Wrote {tables_dir}/statistical_test_family.csv ({len(df)} rows: {n_bootstrap} bootstrap + {n_jackknife} jackknife)")

    # --- Compact summary: one row per question type, for the manuscript's in-text table ---
    summary_rows = [
        {
            "Question": "Q1a: Is ROC-AUC distinguishable from chance?", "Null hypothesis": "ROC-AUC = 0.5",
            "Metric": "ROC-AUC", "Comparison": "6 models x 2 negative sets (N0, N2)", "N tests": 12,
            "Test": "Percentile cluster bootstrap / delete-1-cluster jackknife", "Correction": "BH, within each method's 44-test family",
        },
        {
            "Question": "Q1b: Is MCC distinguishable from chance?", "Null hypothesis": "MCC = 0",
            "Metric": "MCC", "Comparison": "6 models x 2 negative sets (N0, N2)", "N tests": 12,
            "Test": "Percentile cluster bootstrap / delete-1-cluster jackknife", "Correction": "BH, within each method's 44-test family",
        },
        {
            "Question": "Q1c: Is PR-AUC distinguishable from chance?", "Null hypothesis": "PR-AUC = test-set prevalence",
            "Metric": "PR-AUC", "Comparison": "6 models x 2 negative sets (N0, N2)", "N tests": 12,
            "Test": "Percentile cluster bootstrap / delete-1-cluster jackknife", "Correction": "BH, within each method's 44-test family",
        },
        {
            "Question": "Q2: Does architecture beat baseline?", "Null hypothesis": "ΔROC-AUC (Siamese MLP - LR) = 0",
            "Metric": "ΔROC-AUC", "Comparison": "Siamese MLP vs. LR, same test set, 2 negative sets", "N tests": 2,
            "Test": "Paired percentile bootstrap / paired jackknife", "Correction": "BH, within each method's 44-test family",
        },
        {
            "Question": "Q3: Does the N0-vs-N2 gap differ from zero?", "Null hypothesis": "ROC-AUC(N0) - ROC-AUC(N2) = 0",
            "Metric": "ΔROC-AUC", "Comparison": "6 models, independent (unpaired) N0 vs. N2 test sets", "N tests": 6,
            "Test": "Unpaired percentile bootstrap / unpaired jackknife (Welch-Satterthwaite)", "Correction": "BH, within each method's 44-test family",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "statistical_test_family_summary.csv", index=False)
    logger.info(f"Wrote {tables_dir}/statistical_test_family_summary.csv ({len(summary_df)} rows, sums to 12+12+12+2+6=44 tests/method)")


if __name__ == "__main__":
    main()
