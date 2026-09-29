"""Multi-seed robustness check for the architecture comparison.

Objective: deep-learning results should not depend on a single seed, so the
most important comparisons use multiple independent seeds while the
underlying data partition is kept fixed. The architecture and statistical
analyses' headline finding -- every architecture collapses to
indistinguishable-from-chance under N2 on R3, and even the N0 "architecture
edge" doesn't survive a paired significance test -- was established from a
single seed (42) for both data construction and model training. This
answers the open question that raised: is that finding itself an artifact
of one lucky/unlucky seed?

Design: the R3 train/test protein-pool partition and N2's degree-matched
negatives are built ONCE with DATA_SEED=42 (identical to the split generation and architecture ablation/12) and
reused for every run below -- only each model's own training randomness
(sklearn `random_state` / `torch.manual_seed`, and, for the torch
architectures, which pairs land in the internal early-stopping validation
carve-out) varies across MODEL_SEEDS. This is deliberately NOT "5 fresh
R3 splits": a valid comparison requires identical folds/test data, so only
initialization/training randomness varies here.

MODEL_SEEDS includes 42 so that one run exactly reproduces the architecture ablation's own
numbers -- a free cross-script consistency check, not just a coincidence.

Writes:
  results/architecture_multiseed/<negset>__<model>__seed<N>/
  tables/table6b_architecture_multiseed.csv       (one row per seed)
  tables/table6b_architecture_multiseed_summary.csv  (mean +/- std over 5 seeds)
  figures/figure_multiseed_robustness.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.negative_sampling import _known_pair_set, build_r3_n0_n2_datasets, compute_protein_stats
from src.evaluation.calibration import calibration_metrics
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.models.attention import AttentionFusion
from src.models.classical import build_model
from src.models.interaction import CrossInteractionModule
from src.models.siamese import SharedEncoderFusion
from src.models.torch_utils import predict_proba as torch_predict_proba
from src.models.torch_utils import train_pair_model
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("multiseed_robustness")

DATA_SEED = 42
MODEL_SEEDS = [42, 43, 44, 45, 46]
TEST_FRACTION = 0.20
FUSION = "combined"
# Identical to the architecture-ablation and calibration definitions -- must stay in sync with
# scripts/15_architecture_ablation.py if that ever changes.
CLASSICAL_MODELS = {
    "logistic_regression": {"max_iter": 2000},
    "random_forest": {"n_estimators": 300, "n_jobs": -1},
    "mlp": {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True},
}
TORCH_MODELS = {
    "siamese_mlp": lambda dim: SharedEncoderFusion(dim, hidden_dim=256, latent_dim=128),
    "attention_fusion": lambda dim: AttentionFusion(dim, proj_dim=128),
    "cross_interaction": lambda dim: CrossInteractionModule(dim, interaction_dim=128),
}
MODEL_ORDER = list(CLASSICAL_MODELS) + list(TORCH_MODELS)
NEG_SET_LABELS = {"n0": "N0 (original)", "n2": "N2 (degree-matched)"}


def lookup_embeddings(pairs, protein_features):
    feature_cols = [c for c in protein_features.columns if c != "protein_id"]
    feat_by_id = protein_features.set_index("protein_id")[feature_cols]
    h_a = feat_by_id.loc[pairs["protein_a"]].to_numpy(dtype="float32")
    h_b = feat_by_id.loc[pairs["protein_b"]].to_numpy(dtype="float32")
    return h_a, h_b, feature_cols


def main():
    set_seed(DATA_SEED)
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None,
                         names=["protein_a", "protein_b", "label"])
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=10)
    known_pairs = _known_pair_set(pos)
    protein_features = load_esm2_features("mean", REPO_ROOT / "embeddings" / "esm2")

    # Built ONCE, outside the seed loop: the data partition itself must not vary.
    datasets = build_r3_n0_n2_datasets(protein_stats, known_pairs, proteins, REPO_ROOT, DATA_SEED, TEST_FRACTION)

    out_dir = REPO_ROOT / "results" / "architecture_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_seed in MODEL_SEEDS:
        for neg_name, (train_df, test_df) in datasets.items():
            y_train_raw = train_df["label"].to_numpy()
            y_test = test_df["label"].to_numpy()

            for model_name, params in CLASSICAL_MODELS.items():
                x_train, y_train = build_pair_matrix(train_df, protein_features, fusion=FUSION)
                x_test, _ = build_pair_matrix(test_df, protein_features, fusion=FUSION)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

                model = build_model(model_name, params, model_seed)
                model.fit(x_train, y_train)
                y_prob = model.predict_proba(x_test)[:, 1]
                _record(out_dir, rows, neg_name, model_name, model_seed, y_test, y_prob, len(train_df), len(test_df), test_df)

            train_proteins = pd.unique(pd.concat([train_df["protein_a"], train_df["protein_b"]]))
            feature_cols = [c for c in protein_features.columns if c != "protein_id"]
            scaler = StandardScaler()
            scaler.fit(protein_features.set_index("protein_id").loc[train_proteins, feature_cols])
            scaled_features = protein_features.copy()
            scaled_features[feature_cols] = scaler.transform(protein_features[feature_cols])

            h_a_train_s, h_b_train_s, _ = lookup_embeddings(train_df, scaled_features)
            h_a_test_s, h_b_test_s, _ = lookup_embeddings(test_df, scaled_features)

            for model_name, model_fn in TORCH_MODELS.items():
                model = train_pair_model(
                    lambda mf=model_fn: mf(len(feature_cols)),
                    h_a_train_s, h_b_train_s, y_train_raw.astype("float32"), seed=model_seed,
                )
                y_prob = torch_predict_proba(model, h_a_test_s, h_b_test_s)
                _record(out_dir, rows, neg_name, model_name, model_seed, y_test, y_prob, len(train_df), len(test_df), test_df)

        logger.info(f"Completed model_seed={model_seed}")

    per_seed_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    per_seed_df.to_csv(tables_dir / "table6b_architecture_multiseed.csv", index=False)
    logger.info(f"Wrote {tables_dir}/table6b_architecture_multiseed.csv")

    summary = (
        per_seed_df.groupby(["negative_set", "negative_set_label", "model"])
        .agg(
            roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
            mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std"),
            n_seeds=("roc_auc", "count"),
        )
        .reset_index()
    )
    summary.to_csv(tables_dir / "table6b_architecture_multiseed_summary.csv", index=False)
    logger.info(f"Wrote {tables_dir}/table6b_architecture_multiseed_summary.csv")

    _plot_robustness(summary, REPO_ROOT / "figures" / "figure_multiseed_robustness.pdf")
    logger.info(f"Wrote {REPO_ROOT}/figures/figure_multiseed_robustness.pdf")

    write_manifest(out_dir, config={"models": MODEL_ORDER, "negative_sets": ["n0", "n2"], "split": "both_unseen",
                                     "data_seed": DATA_SEED, "model_seeds": MODEL_SEEDS}, seed=DATA_SEED)
    logger.info("Multi-seed robustness check complete.")


def _record(out_dir, rows, neg_name, model_name, model_seed, y_test, y_prob, n_train, n_test, test_df):
    m = classification_metrics(y_test, y_prob)
    c = calibration_metrics(y_test, y_prob)
    combo_dir = out_dir / f"{neg_name}__{model_name}__seed{model_seed}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "y_true": y_test, "y_probability": y_prob,
        "protein_a": test_df["protein_a"].to_numpy(), "protein_b": test_df["protein_b"].to_numpy(),
    })
    predictions.to_csv(combo_dir / "predictions.csv", index=False)
    save_json({**m, **c, "n_train": n_train, "n_test": n_test, "model_seed": model_seed}, combo_dir / "metrics.json")
    write_manifest(combo_dir, config={"negative_set": neg_name, "model": model_name, "model_seed": model_seed}, seed=model_seed)
    rows.append({"negative_set": neg_name, "negative_set_label": NEG_SET_LABELS[neg_name],
                 "model": model_name, "model_seed": model_seed, "n_train": n_train, "n_test": n_test, **m, **c})


def _plot_robustness(summary, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(MODEL_ORDER))
    for i, neg_name in enumerate(["n0", "n2"]):
        sub = summary[summary["negative_set"] == neg_name].set_index("model").loc[MODEL_ORDER]
        offset = (i - 0.5) * 0.35
        ax.bar([xi + offset for xi in x], sub["roc_auc_mean"], width=0.35,
               yerr=sub["roc_auc_std"], capsize=3, label=NEG_SET_LABELS[neg_name])
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("ROC-AUC (mean +/- std over 5 seeds)")
    ax.set_title("Multi-seed robustness: R3 architecture ablation (5 training seeds, fixed data partition)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
