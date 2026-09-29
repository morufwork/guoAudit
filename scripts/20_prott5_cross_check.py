"""ProtT5 cross-check of the negative-sampling confound finding.

Objective: the negative-bias re-validation (scripts/14) and the architecture ablation and statistical analysis
established that this benchmark's apparent R0->R3 generalization gap and
ESM-2-vs-AAC/CTD advantage collapse under degree-matched (N2) or hard (N5)
negatives -- but every one of those results used ESM-2 as the sole PLM.
Is the confound (and its effect on representation comparisons) an ESM-2-
specific artifact, or does it hold for a second, architecturally
independent PLM family (a T5 encoder vs. ESM's BERT-style encoder)?

Design: byte-for-bit the same methodology as scripts/14
(negative_bias_revalidation) -- same dataset-construction functions
(src.data.negative_sampling.build_r0_n0_n2_n5_datasets /
build_r3_n0_n2_n5_datasets), same MLP classifier, same fusion, same splits
-- with `prott5_mean` added as a third representation alongside AAC+CTD and
ESM-2 mean. Re-deriving the AAC+CTD/ESM-2 numbers here (rather than only
adding ProtT5) is deliberate: it puts all three representations in one
directly comparable table, and doubles as a regression check that this
script's harness reproduces scripts/14's already-verified numbers exactly.

Writes:
  results/prott5_cross_check/<split>__<negset>__<representation>/
  tables/prott5_cross_check_summary.csv
  figures/prott5_cross_check.pdf
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
from src.features.prott5 import load_prott5_features
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("prott5_cross_check")

SEED = 42
TEST_FRACTION = 0.20
MLP_PARAMS = {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True}
FUSION = "combined"
NEG_SET_LABELS = {"n0": "N0 (original)", "n2": "N2 (degree-matched)", "n5": "N5 (hard)"}
REPRESENTATIONS = ["aac_ctd", "esm2_mean", "prott5_mean"]


def load_representation(name: str) -> pd.DataFrame:
    if name == "esm2_mean":
        return load_esm2_features("mean", REPO_ROOT / "embeddings" / "esm2")
    if name == "prott5_mean":
        return load_prott5_features(REPO_ROOT / "embeddings" / "prott5")
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

    out_dir = REPO_ROOT / "results" / "prott5_cross_check"
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
    results_df.to_csv(tables_dir / "prott5_cross_check_summary.csv", index=False)
    logger.info(f"Wrote {tables_dir}/prott5_cross_check_summary.csv")

    # --- Figure: ROC-AUC by negative set, split into R0/R3 panels, representation as color ---
    figures_dir = REPO_ROOT / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"aac_ctd": "#1b9e77", "esm2_mean": "#d95f02", "prott5_mean": "#7570b3"}
    neg_order = ["n0", "n2", "n5"]

    for ax, split_name in zip(axes, ["random", "both_unseen"]):
        x = range(len(neg_order))
        for i, repr_name in enumerate(REPRESENTATIONS):
            sub = results_df[(results_df["split"] == split_name) & (results_df["representation"] == repr_name)]
            sub = sub.set_index("negative_set").loc[neg_order]
            offset = (i - 1) * 0.27
            ax.bar([xi + offset for xi in x], sub["roc_auc"], width=0.27, label=repr_name, color=colors[repr_name])
        ax.set_xticks(list(x))
        ax.set_xticklabels([NEG_SET_LABELS[n].split(" (")[0] for n in neg_order])
        ax.set_title(split_labels[split_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(fontsize=8)
    fig.suptitle("ProtT5 cross-check: does the confound generalize across PLM families? (MLP, consistent negatives)")
    fig.tight_layout()
    fig.savefig(figures_dir / "prott5_cross_check.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/prott5_cross_check.pdf")

    write_manifest(out_dir, config={"splits": list(all_datasets), "negative_sets": neg_order, "representations": REPRESENTATIONS}, seed=SEED)
    logger.info("ProtT5 cross-check complete.")


if __name__ == "__main__":
    main()
