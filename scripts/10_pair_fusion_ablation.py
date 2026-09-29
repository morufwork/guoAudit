"""Pair-fusion ablation.

Compares how two ESM-2 (mean-pooled, the representation comparison's recommended representation)
protein embeddings should be combined: concatenation, absolute difference,
Hadamard product, sum, diff+product, sum+diff+product ("combined"), and a
learned symmetric fusion (src/models/siamese.py).

Evaluated on R0 (naive) and R3 (protein-disjoint) -- directly tests the
concern that concatenation is order-sensitive and, under a
naive split where the same protein recurs across train/test, may look
artificially strong at R0 in a way that collapses under R3. This is the
same hypothesis flagged (not asserted) back in the conventional baseline.

Same MLP classifier for every fixed-fusion candidate (isolates the fusion
effect); the learned-fusion candidate necessarily trains its own encoder
end-to-end since the fusion IS the learned part.

Writes:
  results/pair_fusion_ablation/<split>__<fusion>/
  tables/pair_fusion_ablation.csv
  figures/pair_fusion_performance.pdf
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

from src.evaluation.calibration import calibration_metrics
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.features.pair_fusion import ABLATION_FUSIONS
from src.models.classical import build_model
from src.models.siamese import predict_proba, train_shared_encoder_fusion
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("pair_fusion_ablation")

SEED = 42
MLP_PARAMS = {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True}
FIXED_FUSIONS = ["sum", "abs_diff", "hadamard", "diff_product", "combined", "concat"]
SPLITS = [("random", "Random (R0)", 0), ("both_unseen", "Both-unseen (R3)", 1)]


def eval_fixed_fusion(fusion_name, train_pairs, test_pairs, protein_features):
    x_train, y_train = build_pair_matrix(train_pairs, protein_features, fusion=fusion_name)
    x_test, y_test = build_pair_matrix(test_pairs, protein_features, fusion=fusion_name)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = build_model("mlp", MLP_PARAMS, SEED)
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    return y_test, y_prob


def lookup_embeddings(pairs, protein_features):
    feature_cols = [c for c in protein_features.columns if c != "protein_id"]
    feat_by_id = protein_features.set_index("protein_id")[feature_cols]
    h_a = feat_by_id.loc[pairs["protein_a"]].to_numpy(dtype=np.float64)
    h_b = feat_by_id.loc[pairs["protein_b"]].to_numpy(dtype=np.float64)
    return h_a, h_b


def eval_learned_symmetric(train_pairs, test_pairs, protein_features):
    feature_cols = [c for c in protein_features.columns if c != "protein_id"]
    train_protein_ids = pd.unique(pd.concat([train_pairs["protein_a"], train_pairs["protein_b"]]))
    scaler = StandardScaler()
    scaler.fit(protein_features.set_index("protein_id").loc[train_protein_ids, feature_cols])

    scaled_features = protein_features.copy()
    scaled_features[feature_cols] = scaler.transform(protein_features[feature_cols])

    h_a_train, h_b_train = lookup_embeddings(train_pairs, scaled_features)
    h_a_test, h_b_test = lookup_embeddings(test_pairs, scaled_features)
    y_train = train_pairs["label"].to_numpy().astype(np.float32)
    y_test = test_pairs["label"].to_numpy()

    model = train_shared_encoder_fusion(h_a_train, h_b_train, y_train, input_dim=len(feature_cols), seed=SEED)
    y_prob = predict_proba(model, h_a_test, h_b_test)

    # Pair-order invariance check: must hold
    # exactly, not approximately, since fusion is a commutative sum.
    y_prob_swapped = predict_proba(model, h_b_test, h_a_test)
    max_diff = float(np.max(np.abs(y_prob - y_prob_swapped)))
    assert max_diff < 1e-5, f"learned symmetric fusion violated pair-order invariance: max diff {max_diff}"

    return y_test, y_prob, max_diff


def main():
    set_seed(SEED)
    protein_features = load_esm2_features("mean", REPO_ROOT / "embeddings" / "esm2")

    pair_fusion_ablation_dir = REPO_ROOT / "results" / "pair_fusion_ablation"
    pair_fusion_ablation_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for split_name, split_label, difficulty in SPLITS:
        split_dir = REPO_ROOT / "data" / "splits" / split_name
        train_pairs = pd.read_csv(split_dir / "train.tsv", sep="\t")
        test_pairs = pd.read_csv(split_dir / "test.tsv", sep="\t")

        for fusion_name in FIXED_FUSIONS:
            y_test, y_prob = eval_fixed_fusion(fusion_name, train_pairs, test_pairs, protein_features)
            m = classification_metrics(y_test, y_prob)
            c = calibration_metrics(y_test, y_prob)

            combo_dir = pair_fusion_ablation_dir / f"{split_name}__{fusion_name}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"y_true": y_test, "y_probability": y_prob}).to_csv(combo_dir / "predictions.csv", index=False)
            save_json({**m, **c}, combo_dir / "metrics.json")
            write_manifest(combo_dir, config={"split": split_name, "fusion": fusion_name, "model": "mlp", "params": MLP_PARAMS}, seed=SEED)

            logger.info(f"[{split_label} x {fusion_name}] ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f}")
            rows.append({"split": split_name, "split_label": split_label, "difficulty": difficulty,
                          "fusion": fusion_name, "symmetric": fusion_name != "concat", **m, **c})

        # --- Learned symmetric fusion ---
        y_test, y_prob, max_pair_order_diff = eval_learned_symmetric(train_pairs, test_pairs, protein_features)
        m = classification_metrics(y_test, y_prob)
        c = calibration_metrics(y_test, y_prob)

        combo_dir = pair_fusion_ablation_dir / f"{split_name}__learned_symmetric"
        combo_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"y_true": y_test, "y_probability": y_prob}).to_csv(combo_dir / "predictions.csv", index=False)
        save_json({**m, **c, "max_pair_order_diff": max_pair_order_diff}, combo_dir / "metrics.json")
        write_manifest(combo_dir, config={"split": split_name, "fusion": "learned_symmetric"}, seed=SEED)

        logger.info(f"[{split_label} x learned_symmetric] ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
                    f"MCC={m['mcc']:.4f} (pair-order max diff: {max_pair_order_diff:.2e})")
        rows.append({"split": split_name, "split_label": split_label, "difficulty": difficulty,
                      "fusion": "learned_symmetric", "symmetric": True, **m, **c})

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "pair_fusion_ablation.csv", index=False)
    logger.info(f"Wrote {tables_dir}/pair_fusion_ablation.csv")

    # --- Figure: fusion performance, R0 vs R3 ---
    figures_dir = REPO_ROOT / "figures"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fusion_order = FIXED_FUSIONS + ["learned_symmetric"]
    x = np.arange(len(fusion_order))
    width = 0.35

    for ax, metric, title in zip(axes, ["roc_auc", "pr_auc", "mcc"], ["ROC-AUC", "PR-AUC", "MCC"]):
        for offset, (split_name, split_label, _) in zip([-width / 2, width / 2], SPLITS):
            values = [results_df[(results_df["split"] == split_name) & (results_df["fusion"] == f)][metric].iloc[0] for f in fusion_order]
            colors = ["#d95f02" if f == "concat" else "#1b9e77" for f in fusion_order]
            ax.bar(x + offset, values, width, label=split_label,
                   color=colors if split_name == "both_unseen" else "#7570b3",
                   alpha=1.0 if split_name == "both_unseen" else 0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(fusion_order, rotation=45, ha="right")
        ax.set_title(title)
        ax.axhline(0.5 if metric != "mcc" else 0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)
    fig.suptitle("Pair-fusion ablation: ESM-2 (mean) + MLP, Random (R0) vs Both-unseen (R3)")
    fig.tight_layout()
    fig.savefig(figures_dir / "pair_fusion_performance.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/pair_fusion_performance.pdf")

    write_manifest(pair_fusion_ablation_dir, config={"fusions": fusion_order, "splits": [s[0] for s in SPLITS]}, seed=SEED)
    logger.info("Pair-fusion ablation complete.")


if __name__ == "__main__":
    main()
