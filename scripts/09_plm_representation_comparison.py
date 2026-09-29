"""Controlled PLM representation comparison.

Plan's explicit methodology: "Use the same downstream classifier and same
folds when comparing representations" -- isolates the representation
effect from the architecture effect. MLP is the shared
classifier,
evaluated on the same 8 splits used by scripts/08 (R0-R3 + homology
50/40/30/20%).

AAC+CTD->MLP numbers are NOT retrained here -- they are reused directly
from scripts/08's results/degradation_summary/metrics_summary.csv, since a
valid comparison requires identical splits for a valid comparison and these already are
(same data/splits/ files, same seed). Only ESM-2 (mean and CLS pooling,
both extracted in the ESM-2 extraction) are newly trained here.

Key question: which representation RETAINS performance
as difficulty increases, not just which has the highest random-split score.

Writes:
  results/representation_comparison/<split>__esm2_<pooling>/   (self-contained, Section 13)
  results/representation_comparison_summary/metrics_summary.csv
  tables/table5_plm_representation_comparison.csv
  figures/figure5_representation_comparison.pdf
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
from src.features.esm2 import load_esm2_features
from src.models.classical import build_model
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("plm_representation_comparison")

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

MLP_PARAMS = {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True}
SEED = 42
FUSION = "combined"
POOLINGS = ["mean", "cls"]


def main():
    set_seed(SEED)

    representation_comparison_dir = REPO_ROOT / "results" / "representation_comparison"
    representation_comparison_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    new_predictions = []

    for pooling in POOLINGS:
        protein_features = load_esm2_features(pooling, REPO_ROOT / "embeddings" / "esm2")
        representation = f"esm2_{pooling}"

        for split_name, split_label, difficulty in SPLITS:
            split_dir = REPO_ROOT / "data" / "splits" / split_name
            train_pairs = pd.read_csv(split_dir / "train.tsv", sep="\t")
            test_pairs = pd.read_csv(split_dir / "test.tsv", sep="\t")

            x_train, y_train = build_pair_matrix(train_pairs, protein_features, fusion=FUSION)
            x_test, y_test = build_pair_matrix(test_pairs, protein_features, fusion=FUSION)

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            combo_dir = representation_comparison_dir / f"{split_name}__{representation}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            combo_logger = get_logger(f"representation_comparison.{split_name}.{representation}", log_dir=combo_dir, filename="run.log")

            model = build_model("mlp", MLP_PARAMS, SEED)
            model.fit(x_train_scaled, y_train)
            y_prob = model.predict_proba(x_test_scaled)[:, 1]

            m = classification_metrics(y_test, y_prob)
            c = calibration_metrics(y_test, y_prob)

            predictions = test_pairs[["protein_a", "protein_b"]].assign(
                y_true=y_test, y_probability=y_prob, y_pred=(y_prob >= 0.5).astype(int),
                split=split_name, model="mlp", representation=representation, seed=SEED,
            )
            predictions.to_csv(combo_dir / "predictions.csv", index=False)
            save_json({**m, **c}, combo_dir / "metrics.json")
            write_manifest(combo_dir, config={"split": split_name, "model": "mlp", "params": MLP_PARAMS,
                                               "representation": representation, "fusion": FUSION}, seed=SEED)

            combo_logger.info(f"roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} mcc={m['mcc']:.4f}")
            logger.info(f"[{split_label} x {representation}] n_train={len(train_pairs)} n_test={len(test_pairs)} "
                        f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f}")

            new_rows.append({"split": split_name, "split_label": split_label, "difficulty": difficulty,
                              "representation": representation, "model": "mlp",
                              "n_train": len(train_pairs), "n_test": len(test_pairs), **m, **c})
            new_predictions.append(predictions)

    new_metrics_df = pd.DataFrame(new_rows)

    # --- Reuse the AAC+CTD -> MLP numbers from scripts/08 ---
    gate_b = pd.read_csv(REPO_ROOT / "results" / "degradation_summary" / "metrics_summary.csv")
    aac_ctd_mlp = gate_b[gate_b["model"] == "mlp"].copy()
    aac_ctd_mlp["representation"] = "aac_ctd"
    aac_ctd_mlp = aac_ctd_mlp[new_metrics_df.columns]

    combined = pd.concat([aac_ctd_mlp, new_metrics_df], ignore_index=True).sort_values(["difficulty", "representation"])

    summary_dir = REPO_ROOT / "results" / "representation_comparison_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(summary_dir / "metrics_summary.csv", index=False)
    pd.concat(new_predictions, ignore_index=True).to_csv(summary_dir / "esm2_predictions.csv", index=False)

    tables_dir = REPO_ROOT / "tables"
    combined.to_csv(tables_dir / "table5_plm_representation_comparison.csv", index=False)
    logger.info(f"Wrote {tables_dir}/table5_plm_representation_comparison.csv")

    # --- Figure 5: representation comparison across strict evaluation regimes ---
    figures_dir = REPO_ROOT / "figures"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_names = ["roc_auc", "pr_auc", "mcc"]
    metric_titles = ["ROC-AUC", "PR-AUC", "MCC"]
    colors = {"aac_ctd": "#1b9e77", "esm2_mean": "#d95f02", "esm2_cls": "#7570b3"}
    labels = {"aac_ctd": "AAC+CTD", "esm2_mean": "ESM-2 (mean)", "esm2_cls": "ESM-2 (CLS)"}
    x_labels = [label for _, label, _ in SPLITS]

    for ax, metric, title in zip(axes, metric_names, metric_titles):
        for repr_name in ["aac_ctd", "esm2_mean", "esm2_cls"]:
            sub = combined[combined["representation"] == repr_name].sort_values("difficulty")
            ax.plot(sub["difficulty"], sub[metric], marker="o", label=labels[repr_name], color=colors[repr_name])
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Generalization difficulty")
        ax.axhline(0.5 if metric != "mcc" else 0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Representation comparison (shared classifier: MLP, single split per point, seed 42)")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure5_representation_comparison.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/figure5_representation_comparison.pdf")

    write_manifest(summary_dir, config={"splits": [s[0] for s in SPLITS], "representations": ["aac_ctd", "esm2_mean", "esm2_cls"], "model": "mlp"}, seed=SEED)
    logger.info("PLM representation comparison complete.")


if __name__ == "__main__":
    main()
