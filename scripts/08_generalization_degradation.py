"""Generalization degradation curve across protein-novelty and sequence-similarity splits.

Trains the conventional baseline models (Logistic Regression, Random Forest,
XGBoost, MLP) on the AAC+CTD representation across every split
built by scripts 03 and 04: Random(R0), Seen-seen(R1), One-unseen(R2),
Both-unseen(R3), and homology-controlled 50/40/30/20%.

Each split is evaluated ONCE (train.tsv -> test.tsv), not via CV -- these
are explicit fixed splits, never regenerated implicitly.
StandardScaler fit on each split's training data only. Threshold fixed at
0.5. Same 4 models as the conventional baseline, so results are directly comparable to the
the conventional-CV numbers for the R0 column.

Writes:
  results/degradation/<split>__<model>/   (self-contained, Section 13 standard)
  results/degradation_summary/all_predictions.csv
  results/degradation_summary/metrics_summary.csv
  tables/table3_protein_generalization.csv   (R0-R3)
  tables/table4_homology_controlled.csv      (50/40/30/20%)
  figures/figure4_generalization_degradation.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.calibration import calibration_metrics
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("generalization_degradation")

# split_dir_name -> (display label, difficulty rank for plotting)
SPLITS = [
    ("random", "Random (R0)", 0),
    ("seen_seen", "Seen-seen (R1)", 1),
    ("one_unseen", "One-unseen (R2)", 2),
    ("both_unseen", "Both-unseen (R3)", 3),
    ("homology_50", "50% identity", 4),
    ("homology_40", "40% identity", 5),
    ("homology_30", "30% identity", 6),
    ("homology_20", "20% identity", 7),
]

MODELS = {
    "logistic_regression": {"max_iter": 2000},
    "random_forest": {"n_estimators": 300, "n_jobs": -1},
    "xgboost": {"n_estimators": 300, "eval_metric": "logloss"},
    "mlp": {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True},
}

SEED = 42
REPRESENTATION = "aac_ctd"
FUSION = "combined"


def load_aac_ctd_features() -> pd.DataFrame:
    features_dir = REPO_ROOT / "data" / "processed" / "features"
    aac = pd.read_csv(features_dir / "aac.csv")
    ctd = pd.read_csv(features_dir / "ctd.csv")
    return aac.merge(ctd, on="protein_id")


def main():
    set_seed(SEED)
    protein_features = load_aac_ctd_features()

    degradation_dir = REPO_ROOT / "results" / "degradation"
    degradation_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    metrics_rows = []

    for split_name, split_label, difficulty in SPLITS:
        split_dir = REPO_ROOT / "data" / "splits" / split_name
        train_pairs = pd.read_csv(split_dir / "train.tsv", sep="\t")
        test_pairs = pd.read_csv(split_dir / "test.tsv", sep="\t")

        x_train, y_train = build_pair_matrix(train_pairs, protein_features, fusion=FUSION)
        x_test, y_test = build_pair_matrix(test_pairs, protein_features, fusion=FUSION)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        for model_name, params in MODELS.items():
            combo_dir = degradation_dir / f"{split_name}__{model_name}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            combo_logger = get_logger(f"degradation.{split_name}.{model_name}", log_dir=combo_dir, filename="run.log")

            model = build_model(model_name, params, SEED)
            model.fit(x_train_scaled, y_train)
            y_prob = model.predict_proba(x_test_scaled)[:, 1]

            m = classification_metrics(y_test, y_prob)
            c = calibration_metrics(y_test, y_prob)

            predictions = test_pairs[["protein_a", "protein_b"]].assign(
                y_true=y_test,
                y_probability=y_prob,
                y_pred=(y_prob >= 0.5).astype(int),
                split=split_name,
                model=model_name,
                representation=REPRESENTATION,
                seed=SEED,
            )
            predictions.to_csv(combo_dir / "predictions.csv", index=False)
            save_json({**m, **c}, combo_dir / "metrics.json")
            write_manifest(
                combo_dir,
                config={"split": split_name, "model": model_name, "params": params, "representation": REPRESENTATION, "fusion": FUSION},
                seed=SEED,
            )

            combo_logger.info(f"roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} mcc={m['mcc']:.4f} "
                               f"brier={c['brier_score']:.4f} ece={c['ece']:.4f}")
            logger.info(f"[{split_label} x {model_name}] n_train={len(train_pairs)} n_test={len(test_pairs)} "
                        f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f}")

            all_predictions.append(predictions)
            metrics_rows.append({
                "split": split_name, "split_label": split_label, "difficulty": difficulty,
                "model": model_name, "n_train": len(train_pairs), "n_test": len(test_pairs),
                **m, **c,
            })

    # --- Aggregate ---
    summary_dir = REPO_ROOT / "results" / "degradation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_csv(summary_dir / "all_predictions.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["difficulty", "model"])
    metrics_df.to_csv(summary_dir / "metrics_summary.csv", index=False)

    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    protein_gen = metrics_df[metrics_df["split"].isin(["random", "seen_seen", "one_unseen", "both_unseen"])]
    protein_gen.to_csv(tables_dir / "table3_protein_generalization.csv", index=False)
    homology = metrics_df[metrics_df["split"].str.startswith("homology_")]
    homology.to_csv(tables_dir / "table4_homology_controlled.csv", index=False)
    logger.info(f"Wrote {tables_dir}/table3_protein_generalization.csv and table4_homology_controlled.csv")

    # --- Figure 4: generalization degradation curve ---
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_names = ["roc_auc", "pr_auc", "mcc"]
    metric_titles = ["ROC-AUC", "PR-AUC", "MCC"]
    colors = {"logistic_regression": "#7570b3", "random_forest": "#1b9e77", "xgboost": "#d95f02", "mlp": "#e7298a"}
    x_labels = [label for _, label, _ in SPLITS]

    for ax, metric, title in zip(axes, metric_names, metric_titles):
        for model_name in MODELS:
            sub = metrics_df[metrics_df["model"] == model_name].sort_values("difficulty")
            ax.plot(sub["difficulty"], sub[metric], marker="o", label=model_name, color=colors[model_name])
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Generalization difficulty")
        ax.axhline(0.5 if metric != "mcc" else 0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Generalization degradation: AAC+CTD + classical ML (single split per point, seed 42)")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure4_generalization_degradation.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/figure4_generalization_degradation.pdf")

    write_manifest(summary_dir, config={"splits": [s[0] for s in SPLITS], "models": list(MODELS), "representation": REPRESENTATION}, seed=SEED)
    logger.info("Generalization degradation analysis complete.")


if __name__ == "__main__":
    main()
