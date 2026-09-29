"""Model architecture study.

Objective: does model complexity provide additional value once
representation and split behavior are established?
Per Decision Gate 2, this question is only worth asking if simpler models
leave a clear performance/generalization gap.

Given the negative-bias re-validation's finding (N0's degree confound was
the dominant signal source in prior evaluations), this run does NOT use N0
alone -- it compares every model under both N0 (reference) and N2
(degree-matched, confound-controlled), on R3 (both-unseen, the strictest
split), ESM-2 mean pooling. This directly tests whether added architecture
capacity can recover signal N2 hides from simpler models, or whether R3/N2
performance is capped near chance regardless of model complexity.

Models, progressively more complex:
  1. Logistic Regression
  2. Random Forest
  3. MLP
  4. Siamese MLP (src/models/siamese.py, reused from the pair-fusion ablation's
     learned_symmetric -- same architecture, new role)
  5. Attention-based fusion (src/models/attention.py)
  6. Cross-protein interaction module (src/models/interaction.py)

Writes:
  results/architecture_ablation/<negset>__<model>/
  tables/table6_architecture_ablation.csv
  figures/architecture_ablation.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("architecture_ablation")

SEED = 42
TEST_FRACTION = 0.20
FUSION = "combined"
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
NEG_SET_LABELS = {"n0": "N0 (original)", "n2": "N2 (degree-matched)"}


def lookup_embeddings(pairs, protein_features):
    feature_cols = [c for c in protein_features.columns if c != "protein_id"]
    feat_by_id = protein_features.set_index("protein_id")[feature_cols]
    h_a = feat_by_id.loc[pairs["protein_a"]].to_numpy(dtype="float32")
    h_b = feat_by_id.loc[pairs["protein_b"]].to_numpy(dtype="float32")
    return h_a, h_b, feature_cols


def main():
    set_seed(SEED)
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=10)
    known_pairs = _known_pair_set(pos)
    protein_features = load_esm2_features("mean", REPO_ROOT / "embeddings" / "esm2")

    datasets = build_r3_n0_n2_datasets(protein_stats, known_pairs, proteins, REPO_ROOT, SEED, TEST_FRACTION)

    out_dir = REPO_ROOT / "results" / "architecture_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for neg_name, (train_df, test_df) in datasets.items():
        h_a_train, h_b_train, feature_cols = lookup_embeddings(train_df, protein_features)
        h_a_test, h_b_test, _ = lookup_embeddings(test_df, protein_features)
        y_train_raw = train_df["label"].to_numpy()
        y_test = test_df["label"].to_numpy()

        # --- Classical models: use the fixed "combined" fusion vector ---
        for model_name, params in CLASSICAL_MODELS.items():
            x_train, y_train = build_pair_matrix(train_df, protein_features, fusion=FUSION)
            x_test, _ = build_pair_matrix(test_df, protein_features, fusion=FUSION)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            model = build_model(model_name, params, SEED)
            model.fit(x_train, y_train)
            y_prob = model.predict_proba(x_test)[:, 1]
            _record(out_dir, rows, neg_name, model_name, y_test, y_prob, len(train_df), len(test_df), test_df)

        # --- Torch architectures: operate on raw (unfused) embeddings, scaled per-protein ---
        train_proteins = pd.unique(pd.concat([train_df["protein_a"], train_df["protein_b"]]))
        scaler = StandardScaler()
        feat_by_id = protein_features.set_index("protein_id")[feature_cols]
        scaler.fit(feat_by_id.loc[train_proteins])
        scaled_features = protein_features.copy()
        scaled_features[feature_cols] = scaler.transform(protein_features[feature_cols])

        h_a_train_s, h_b_train_s, _ = lookup_embeddings(train_df, scaled_features)
        h_a_test_s, h_b_test_s, _ = lookup_embeddings(test_df, scaled_features)

        for model_name, model_fn in TORCH_MODELS.items():
            model = train_pair_model(
                lambda mf=model_fn: mf(len(feature_cols)),
                h_a_train_s, h_b_train_s, y_train_raw.astype("float32"), seed=SEED,
            )
            y_prob = torch_predict_proba(model, h_a_test_s, h_b_test_s)
            _record(out_dir, rows, neg_name, model_name, y_test, y_prob, len(train_df), len(test_df), test_df)

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    results_df.to_csv(tables_dir / "table6_architecture_ablation.csv", index=False)
    logger.info(f"Wrote {tables_dir}/table6_architecture_ablation.csv")

    figures_dir = REPO_ROOT / "figures"
    model_order = list(CLASSICAL_MODELS) + list(TORCH_MODELS)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(model_order))
    for i, neg_name in enumerate(["n0", "n2"]):
        sub = results_df[results_df["negative_set"] == neg_name].set_index("model").loc[model_order]
        offset = (i - 0.5) * 0.35
        ax.bar([xi + offset for xi in x], sub["roc_auc"], width=0.35, label=NEG_SET_LABELS[neg_name])
    ax.set_xticks(list(x))
    ax.set_xticklabels(model_order, rotation=30, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Architecture ablation: Both-unseen (R3), ESM-2 mean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "architecture_ablation.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/architecture_ablation.pdf")

    write_manifest(out_dir, config={"models": model_order, "negative_sets": ["n0", "n2"], "split": "both_unseen"}, seed=SEED)
    logger.info("Architecture ablation complete.")


def _record(out_dir, rows, neg_name, model_name, y_test, y_prob, n_train, n_test, test_df=None):
    m = classification_metrics(y_test, y_prob)
    c = calibration_metrics(y_test, y_prob)
    combo_dir = out_dir / f"{neg_name}__{model_name}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    predictions = {"y_true": y_test, "y_probability": y_prob}
    if test_df is not None:
        predictions["protein_a"] = test_df["protein_a"].to_numpy()
        predictions["protein_b"] = test_df["protein_b"].to_numpy()
    pd.DataFrame(predictions).to_csv(combo_dir / "predictions.csv", index=False)
    save_json({**m, **c, "n_train": n_train, "n_test": n_test}, combo_dir / "metrics.json")
    write_manifest(combo_dir, config={"negative_set": neg_name, "model": model_name}, seed=SEED)
    logger.info(f"[{NEG_SET_LABELS[neg_name]} x {model_name}] n_train={n_train} n_test={n_test} "
                f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f}")
    rows.append({"negative_set": neg_name, "negative_set_label": NEG_SET_LABELS[neg_name],
                 "model": model_name, "n_train": n_train, "n_test": n_test, **m, **c})


if __name__ == "__main__":
    main()
