"""Degree-computation train/test-label-exposure
sensitivity check.

Every negative-sampling script in this project computes node degree
(`compute_protein_stats`) from the FULL positive-pair set -- train and test
positives combined -- before any split is applied (see e.g.
`build_r0_n0_n2_n5_datasets`, which receives one `protein_stats` object
built from all of `pos`). For R0 (the naive random pair split), that means
a test pair's N2 decoy is matched against a degree value that can itself be
informed by that same protein's OTHER positive edges landing in the test
set -- a mild form of label exposure worth
quantifying, even though this study is a retrospective benchmark audit, not
a deployable prospective sampling algorithm (Section 4.3).

This script re-runs R0's N0-vs-N2 comparison (scripts/14's exact template:
same seed, same test fraction, same reduced model set logic) twice, varying
only where `positive_degree` is computed from:
  full_graph  -- current manuscript methodology (train+test positives)
  train_only  -- degree computed from the TRAIN split's positives only

R3 (both-unseen) is deliberately NOT attempted here: by construction, R3
test-pool proteins have zero edges in the training positive graph, so
"train-only degree" for them is not a sensitivity variant -- it is
undefined (uniformly zero). Forcing this experiment for R3 would test
nothing; Section 4.3 characterizes N2 as a post hoc benchmark-control
diagnostic for exactly this reason.

Writes:
  tables/degree_leakage_sensitivity.csv
  tables/degree_leakage_sensitivity_balance.csv
  results/degree_source_sensitivity/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.negative_sampling import _known_pair_set, build_r0_n0_n2_n5_datasets, compute_protein_stats
from src.evaluation.calibration import calibration_metrics
from src.evaluation.degree_distribution import compare_degree_distributions
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.models.classical import build_model
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("degree_leakage_sensitivity")

SEED = 42
TEST_FRACTION = 0.20
N_BOOT = 2000
FUSION = "combined"
MODELS = {
    "logistic_regression": {"max_iter": 2000},
    "mlp": {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True},
}
DEGREE_SOURCE_LABELS = {
    "full_graph": "Full graph (train+test positives; current methodology)",
    "train_only": "Train-only positives (sensitivity variant)",
}
NEG_SET_LABELS = {"n0": "N0 (original)", "n2": "N2 (degree-matched)"}


def main():
    set_seed(SEED)
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None,
                         names=["protein_a", "protein_b", "label"])
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)
    known_pairs = _known_pair_set(pos)

    # Reproduces the exact split `build_r0_n0_n2_n5_datasets` performs
    # internally (same seed/fraction), so `pos_train` here is identical to
    # what that function uses -- needed to compute train-only degree
    # *before* calling it, without modifying the shared builder.
    pos_train, _pos_test = train_test_split(pos, test_size=TEST_FRACTION, random_state=SEED)

    degree_sources = {
        "full_graph": compute_protein_stats(proteins, pos, n_deciles=10),
        "train_only": compute_protein_stats(proteins, pos_train, n_deciles=10),
    }

    protein_features = load_esm2_features("mean", REPO_ROOT / "embeddings" / "esm2")

    out_dir = REPO_ROOT / "results" / "degree_source_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    balance_rows = []

    for source_name, protein_stats in degree_sources.items():
        n_zero_degree = int((protein_stats["positive_degree"] == 0).sum())
        logger.info(f"=== degree source: {source_name} ({n_zero_degree}/{len(protein_stats)} proteins at degree 0) ===")

        datasets = build_r0_n0_n2_n5_datasets(pos, neg0, protein_stats, known_pairs, SEED, TEST_FRACTION)

        # Balance diagnostic for
        # this degree source's N2, to check whether train-only degree also
        # degrades matching quality, not just downstream model performance.
        n2_train, n2_test = datasets["n2"]
        n2_all = pd.concat([n2_train[n2_train["label"] == 0], n2_test[n2_test["label"] == 0]], ignore_index=True)
        balance = compare_degree_distributions(pos, n2_all, protein_stats["positive_degree"], N_BOOT, SEED)
        logger.info(f"[{source_name}] positive_vs_n2 SMD={balance['standardized_mean_difference']:.3f} "
                    f"(well_balanced<0.1: {balance['smd_well_balanced']}) KS={balance['ks_statistic']:.3f}")
        balance_rows.append({"degree_source": source_name, "degree_source_label": DEGREE_SOURCE_LABELS[source_name], **balance})

        for neg_name in ["n0", "n2"]:
            train_df, test_df = datasets[neg_name]
            x_train, y_train = build_pair_matrix(train_df, protein_features, fusion=FUSION)
            x_test, y_test = build_pair_matrix(test_df, protein_features, fusion=FUSION)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            for model_name, params in MODELS.items():
                model = build_model(model_name, params, SEED)
                model.fit(x_train, y_train)
                y_prob = model.predict_proba(x_test)[:, 1]

                m = classification_metrics(y_test, y_prob)
                c = calibration_metrics(y_test, y_prob)
                combo_dir = out_dir / f"{source_name}__{neg_name}__{model_name}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                save_json({**m, **c, "n_train": len(train_df), "n_test": len(test_df)}, combo_dir / "metrics.json")
                write_manifest(combo_dir, config={"degree_source": source_name, "negative_set": neg_name, "model": model_name}, seed=SEED)

                logger.info(f"[{source_name} x {NEG_SET_LABELS[neg_name]} x {model_name}] "
                            f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f}")
                rows.append({
                    "degree_source": source_name, "degree_source_label": DEGREE_SOURCE_LABELS[source_name],
                    "negative_set": neg_name, "negative_set_label": NEG_SET_LABELS[neg_name],
                    "model": model_name, "n_train": len(train_df), "n_test": len(test_df), **m, **c,
                })

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    results_df.to_csv(tables_dir / "degree_leakage_sensitivity.csv", index=False)
    logger.info(f"Wrote {tables_dir}/degree_leakage_sensitivity.csv")

    balance_df = pd.DataFrame(balance_rows)
    balance_df.to_csv(tables_dir / "degree_leakage_sensitivity_balance.csv", index=False)
    logger.info(f"Wrote {tables_dir}/degree_leakage_sensitivity_balance.csv")

    # --- Does the N0-vs-N2 gap survive train-only degree? ---
    gap_summary = []
    for source_name in degree_sources:
        for model_name in MODELS:
            sub = results_df[(results_df["degree_source"] == source_name) & (results_df["model"] == model_name)].set_index("negative_set")
            gap_summary.append({
                "degree_source": source_name, "model": model_name,
                "roc_auc_n0": float(sub.loc["n0", "roc_auc"]), "roc_auc_n2": float(sub.loc["n2", "roc_auc"]),
                "gap": float(sub.loc["n0", "roc_auc"] - sub.loc["n2", "roc_auc"]),
            })
    gap_df = pd.DataFrame(gap_summary)
    logger.info(f"N0-vs-N2 gap by degree source:\n{gap_df.to_string(index=False)}")

    save_json(
        {
            "note": (
                "Sensitivity check for whether N2's degree computation "
                "exposes test-set positive labels. R0 only -- R3 (both-unseen) is not attempted "
                "because R3 test-pool proteins have zero edges in the training positive graph by "
                "construction, so train-only degree is undefined (uniformly zero) for them, not a "
                "meaningful sensitivity variant. See Section 4.3 for the resulting scope statement."
            ),
            "gap_summary": gap_summary,
            "collapse_survives_train_only_degree": bool((gap_df["roc_auc_n2"] < 0.55).all()),
        },
        out_dir / "degree_leakage_sensitivity_summary.json",
    )
    write_manifest(out_dir, config={"degree_sources": list(degree_sources), "models": list(MODELS), "split": "random"}, seed=SEED,
                    extra={"script": "37_degree_leakage_sensitivity.py"})
    logger.info("Degree leakage sensitivity check complete.")


if __name__ == "__main__":
    main()
