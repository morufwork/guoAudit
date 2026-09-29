"""Negative-sampling bias re-validation.

the negative-sampling study found that a model trained/evaluated on the benchmark's own
negatives (N0) collapses to chance against degree-matched negatives (N2)
or hard negatives (N5) -- but that experiment swapped negatives at
train/test time, deliberately creating a mismatch. It did NOT test whether
the R0->R3 generalization gap and the ESM-2-vs-AAC/CTD advantage still hold when a negative set is used *consistently* (same
negatives for training and testing, just like N0's own regime) instead of
the original N0. That is what this script resolves.

For each negative set x in {n0, n2, n5} and each split regime in {R0, R3}:
train and evaluate on x-consistent data (never swapped). N2/N5 negatives
for R3 are built respecting the SAME protein-pool partition the split generation used
(candidate_pool restricted to train_pool / test_pool respectively) --
otherwise substituting negatives would silently reopen the leakage that the protein-disjoint split
closed.

Writes:
  results/negative_bias_revalidation/<split>__<negset>__<representation>/
  tables/negative_bias_revalidation_summary.csv
  figures/negative_bias_revalidation.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.negative_sampling import (
    _known_pair_set,
    build_r0_n0_n2_n5_datasets,
    build_r3_n0_n2_n5_datasets,
    compute_protein_stats,
)
from src.evaluation.calibration import calibration_metrics
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("negative_bias_revalidation")

SEED = 42
TEST_FRACTION = 0.20
MLP_PARAMS = {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True}
FUSION = "combined"
NEG_SET_LABELS = {"n0": "N0 (original)", "n2": "N2 (degree-matched)", "n5": "N5 (hard)"}
REPRESENTATIONS = ["aac_ctd", "esm2_mean"]


def load_representation(name: str) -> pd.DataFrame:
    if name == "esm2_mean":
        return load_esm2_features("mean", REPO_ROOT / "embeddings" / "esm2")
    aac = pd.read_csv(REPO_ROOT / "data" / "processed" / "features" / "aac.csv")
    ctd = pd.read_csv(REPO_ROOT / "data" / "processed" / "features" / "ctd.csv")
    return aac.merge(ctd, on="protein_id")


def main():
    set_seed(SEED)
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")

    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=10)
    known_pairs = _known_pair_set(pos)

    logger.info("Building R0 (random) datasets for n0/n2/n5...")
    r0_datasets = build_r0_n0_n2_n5_datasets(pos, neg0, protein_stats, known_pairs, SEED, TEST_FRACTION)
    logger.info("Building R3 (both-unseen) datasets for n0/n2/n5 (protein-pool-restricted decoys)...")
    r3_datasets = build_r3_n0_n2_n5_datasets(protein_stats, known_pairs, proteins, REPO_ROOT, SEED, TEST_FRACTION)

    all_datasets = {"random": r0_datasets, "both_unseen": r3_datasets}
    split_labels = {"random": "Random (R0)", "both_unseen": "Both-unseen (R3)"}

    out_dir = REPO_ROOT / "results" / "negative_bias_revalidation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for split_name, datasets in all_datasets.items():
        for neg_name, (train_df, test_df) in datasets.items():
            for repr_name in REPRESENTATIONS:
                protein_features = load_representation(repr_name)
                x_train, y_train = build_pair_matrix(train_df, protein_features, fusion=FUSION)
                x_test, y_test = build_pair_matrix(test_df, protein_features, fusion=FUSION)

                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

                model = build_model("mlp", MLP_PARAMS, SEED)
                model.fit(x_train, y_train)
                y_prob = model.predict_proba(x_test)[:, 1]

                m = classification_metrics(y_test, y_prob)
                c = calibration_metrics(y_test, y_prob)

                combo_dir = out_dir / f"{split_name}__{neg_name}__{repr_name}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"y_true": y_test, "y_probability": y_prob}).to_csv(combo_dir / "predictions.csv", index=False)
                save_json({**m, **c, "n_train": len(train_df), "n_test": len(test_df)}, combo_dir / "metrics.json")
                write_manifest(combo_dir, config={"split": split_name, "negative_set": neg_name, "representation": repr_name}, seed=SEED)

                logger.info(f"[{split_labels[split_name]} x {NEG_SET_LABELS[neg_name]} x {repr_name}] "
                            f"n_train={len(train_df)} n_test={len(test_df)} "
                            f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f}")
                rows.append({"split": split_name, "split_label": split_labels[split_name],
                             "negative_set": neg_name, "negative_set_label": NEG_SET_LABELS[neg_name],
                             "representation": repr_name, "n_train": len(train_df), "n_test": len(test_df), **m, **c})

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    results_df.to_csv(tables_dir / "negative_bias_revalidation_summary.csv", index=False)
    logger.info(f"Wrote {tables_dir}/negative_bias_revalidation_summary.csv")

    # --- Figure: ROC-AUC by negative set, split into R0/R3 panels, representation as color ---
    figures_dir = REPO_ROOT / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    colors = {"aac_ctd": "#1b9e77", "esm2_mean": "#d95f02"}
    neg_order = ["n0", "n2", "n5"]

    for ax, split_name in zip(axes, ["random", "both_unseen"]):
        x = range(len(neg_order))
        for i, repr_name in enumerate(REPRESENTATIONS):
            sub = results_df[(results_df["split"] == split_name) & (results_df["representation"] == repr_name)]
            sub = sub.set_index("negative_set").loc[neg_order]
            offset = (i - 0.5) * 0.35
            ax.bar([xi + offset for xi in x], sub["roc_auc"], width=0.35, label=repr_name, color=colors[repr_name])
        ax.set_xticks(list(x))
        ax.set_xticklabels([NEG_SET_LABELS[n].split(" (")[0] for n in neg_order])
        ax.set_title(split_labels[split_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(fontsize=8)
    fig.suptitle("Negative-sampling bias re-validation: consistent (not swapped) negatives, MLP")
    fig.tight_layout()
    fig.savefig(figures_dir / "negative_bias_revalidation.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/negative_bias_revalidation.pdf")

    write_manifest(out_dir, config={"splits": list(all_datasets), "negative_sets": neg_order, "representations": REPRESENTATIONS}, seed=SEED)
    logger.info("Negative-sampling bias re-validation complete.")


if __name__ == "__main__":
    main()
