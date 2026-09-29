"""Incremental sequence-vs-degree comparison. as a stronger test than the topology-only baseline
(scripts/24_topology_only_baseline.py) alone.

The topology-only baseline already shows a degree-only classifier
collapses from ROC-AUC 0.962 (N0) to ~0.50 (N2/N5) -- but it evaluates
sequence-only and degree-only models as two entirely separate
experiments. This script instead trains three models on IDENTICAL
train/test splits and negative sets, differing only in their feature
set, so "sequence" vs "degree" vs "sequence+degree" is a genuinely nested
comparison:

  1. sequence-only  -- ESM-2 mean-pooled, "combined" fusion (identical
     representation to Table 7 / scripts/11_negative_sampling_study.py)
  2. degree-only    -- four symmetric functions of positive-degree
     (identical features to scripts/24_topology_only_baseline.py)
  3. sequence+degree -- horizontal concatenation of both

All three use Logistic Regression (the topology-only baseline's primary,
most interpretable model) and reuse the negative-sampling study's exact seed (42), test
fraction (0.20), and saved N2/N5 negative sets, so results are directly
comparable to Table 7 and the topology-only baseline. Only Experiment A
(train on N0, evaluate on N0/N2/N5) is run -- the paper's own
established "primary causal test" (Section 3.5) -- to keep this
addition tightly scoped rather than re-deriving the full A/B matrix.

Writes:
  tables/incremental_sequence_degree.csv
  figures/incremental_sequence_degree.pdf
  results/incremental_sequence_degree/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.negative_sampling import compute_protein_stats
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("incremental_sequence_degree")

NEGATIVE_SET_LABELS = {"n0": "N0 (original)", "n2": "N2 (degree-matched)", "n5": "N5 (degree+length)"}
FEATURE_SETS = ["sequence", "degree", "sequence+degree"]


def build_degree_matrix(pairs_df: pd.DataFrame, degree: pd.Series) -> np.ndarray:
    """Same four symmetric functions of positive-degree as the topology-only
    baseline (sum, |diff|, min, max) -- no sequence content."""
    da = pairs_df["protein_a"].map(degree).to_numpy(dtype=float)
    db = pairs_df["protein_b"].map(degree).to_numpy(dtype=float)
    return np.column_stack([da + db, np.abs(da - db), np.minimum(da, db), np.maximum(da, db)])


def build_features(pairs_df: pd.DataFrame, protein_features: pd.DataFrame, degree: pd.Series, feature_set: str, fusion: str) -> np.ndarray:
    seq_x, _ = build_pair_matrix(pairs_df, protein_features, fusion=fusion)
    deg_x = build_degree_matrix(pairs_df, degree)
    if feature_set == "sequence":
        return seq_x
    if feature_set == "degree":
        return deg_x
    if feature_set == "sequence+degree":
        return np.concatenate([seq_x, deg_x], axis=-1)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def main():
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    config = load_config(REPO_ROOT / "configs" / "hard_negative.yaml")
    seed = config["seed"]
    set_seed(seed)

    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=config["n_deciles"])
    degree = protein_stats["positive_degree"]

    neg_sets_dir = REPO_ROOT / "data" / "processed" / "negative_sets"
    negative_sets = {
        "n0": neg0,
        "n2": pd.read_csv(neg_sets_dir / "n2.tsv", sep="\t"),
        "n5": pd.read_csv(neg_sets_dir / "n5.tsv", sep="\t"),
    }

    # --- Exact reload of the negative-sampling study's split procedure, so results are directly
    # comparable to Table 7 and the topology-only baseline. ---
    pos_train, pos_test = train_test_split(pos, test_size=config["test_fraction"], random_state=seed)
    train_sets, test_sets = {}, {}
    for name, neg_df in negative_sets.items():
        neg_train, neg_test = train_test_split(neg_df, test_size=config["test_fraction"], random_state=seed)
        train_sets[name] = pd.concat([pos_train, neg_train], ignore_index=True)
        test_sets[name] = pd.concat([pos_test, neg_test], ignore_index=True)

    protein_features = load_esm2_features(config["representation"].replace("esm2_", ""), REPO_ROOT / "embeddings" / "esm2")

    rows = []
    for feature_set in FEATURE_SETS:
        logger.info(f"=== feature set: {feature_set} ===")
        x_train = build_features(train_sets["n0"], protein_features, degree, feature_set, config["fusion"])
        y_train = train_sets["n0"]["label"].to_numpy()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = build_model("logistic_regression", {"max_iter": 2000}, seed)
        model.fit(x_train, y_train)

        for name in negative_sets:
            x_test = build_features(test_sets[name], protein_features, degree, feature_set, config["fusion"])
            y_test = test_sets[name]["label"].to_numpy()
            x_test = scaler.transform(x_test)
            y_prob = model.predict_proba(x_test)[:, 1]
            m = classification_metrics(y_test, y_prob)
            logger.info(f"[trained N0 -> tested {name}, {feature_set}] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
            rows.append({"feature_set": feature_set, "trained_on": "n0", "evaluated_on": name, "n_features": x_train.shape[1], **m})

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "incremental_sequence_degree.csv", index=False)
    logger.info(f"Wrote {tables_dir}/incremental_sequence_degree.csv")

    # --- Figure: grouped bars, one group per negative set, one bar per feature set ---
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["n0", "n2", "n5"]
    x = np.arange(len(order))
    width = 0.25
    colors = {"sequence": "#d95f02", "degree": "#1b9e77", "sequence+degree": "#7570b3"}
    for i, feature_set in enumerate(FEATURE_SETS):
        vals = [results_df.set_index(["feature_set", "evaluated_on"]).loc[(feature_set, n), "roc_auc"] for n in order]
        ax.bar(x + (i - 1) * width, vals, width, label=feature_set, color=colors[feature_set])
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([NEGATIVE_SET_LABELS[n] for n in order], rotation=15)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Sequence-only vs. degree-only vs. sequence+degree\n(trained on N0, evaluated on N0/N2/N5)")
    ax.legend()
    fig.tight_layout()
    figures_dir = REPO_ROOT / "figures"
    fig.savefig(figures_dir / "incremental_sequence_degree.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/incremental_sequence_degree.pdf")

    negative_sampling_study_dir = REPO_ROOT / "results" / "incremental_sequence_degree"
    negative_sampling_study_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "note": (
                "Nested sequence-vs-degree comparison "
                "trains Logistic Regression on IDENTICAL N0 train data, varying only the "
                "feature set (sequence-only ESM-2+combined fusion; degree-only 4 symmetric "
                "functions, identical to scripts/24_topology_only_baseline.py; concatenation "
                "of both), evaluated on N0/N2/N5. Reuses the negative-sampling study's exact seed/splits/negative "
                "sets for direct comparability with Table 7 and the topology-only baseline."
            ),
        },
        negative_sampling_study_dir / "incremental_seq_degree_summary.json",
    )
    write_manifest(negative_sampling_study_dir, config=config, seed=seed, extra={"script": "25_incremental_sequence_degree.py"})
    logger.info("Incremental sequence-vs-degree comparison complete.")


if __name__ == "__main__":
    main()
