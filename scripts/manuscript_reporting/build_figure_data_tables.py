"""Rebuild Supplementary Tables S23 and S24 (the data behind main-text Figures 4
and 5) from the committed analysis tables, and check them against the saved files.

  S23 <- tables/table3_protein_generalization.csv + tables/table4_homology_controlled.csv
         (written by scripts/08_generalization_degradation.py)
  S24 <- tables/table5_plm_representation_comparison.csv
         (written by scripts/09_plm_representation_comparison.py)

Usage: python scripts/manuscript_reporting/build_figure_data_tables.py [--write]
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGIME = {"random": "R0", "seen_seen": "R1", "one_unseen": "R2", "both_unseen": "R3",
          "homology_50": "R3-50", "homology_40": "R3-40", "homology_30": "R3-30", "homology_20": "R3-20"}
MODEL = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost", "mlp": "MLP"}
REP = {"aac_ctd": "AAC+CTD", "esm2_mean": "ESM-2 mean", "esm2_cls": "ESM-2 CLS"}
COLS = {"n_train": "Train pairs", "n_test": "Test pairs", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "mcc": "MCC"}


def build(frame, group_col, mapping, label):
    out = frame.assign(Regime=frame["split"].map(REGIME), **{label: frame[group_col].map(mapping)})
    out["_r"] = out["Regime"].map(list(REGIME.values()).index)
    out["_g"] = out[label].map(list(mapping.values()).index)
    return out.sort_values(["_r", "_g"])[["Regime", label, *COLS]].rename(columns=COLS).reset_index(drop=True)


def main():
    t = ROOT / "tables"
    s23 = build(pd.concat([pd.read_csv(t / "table3_protein_generalization.csv"),
                           pd.read_csv(t / "table4_homology_controlled.csv")]), "model", MODEL, "Model")
    t5 = pd.read_csv(t / "table5_plm_representation_comparison.csv")
    s24 = build(t5[t5["model"] == "mlp"], "representation", REP, "Representation")
    for name, df in (("tableS23_figure3_data.csv", s23), ("tableS24_figure4_data.csv", s24)):
        saved = pd.read_csv(t / name)
        pd.testing.assert_frame_equal(df, saved, check_dtype=False)
        print(f"{name}: rebuilt table matches the saved file ({len(df)} rows)")
        if "--write" in sys.argv:
            df.to_csv(t / name, index=False)


if __name__ == "__main__":
    main()
