"""Compact independent-benchmark replication.

Tests whether the yeast benchmark's central diagnostic pattern (Section
3.5-3.6: a degree-only classifier matches or exceeds sequence-based models
on the original negatives; all of them collapse toward chance once
negatives are degree-matched) also appears on a second, independently
constructed PPI benchmark -- the combined C. elegans/Drosophila/E. coli
set (doc/multi_species/, configs/multi_species.yaml). Rather than rerunning the entire
architecture suite, this reuses exactly the R0 (random-split) N0-vs-N2 machinery the main
study already validated (`build_r0_n0_n2_n5_datasets`, dataset-agnostic),
with a reduced model set: a degree-only baseline, Logistic Regression, and
one nonlinear sequence model (MLP), on handcrafted (AAC+CTD) features
always, plus ESM-2-mean features when scripts/41's embeddings are
available (this script degrades gracefully and reports only handcrafted
results if the ESM-2 extraction has not finished yet).

Writes:
  tables/multi_species_negative_sampling.csv
  tables/multi_species_degree_balance.csv
  results/multi_species_pooled/model_comparison.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.negative_sampling import _known_pair_set, build_r0_n0_n2_n5_datasets, compute_protein_stats
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
logger = get_logger("multi_species_model_comparison")

SEED = 42
TEST_FRACTION = 0.20
N_BOOT = 2000
FUSION = "combined"
SEQUENCE_MODELS = {
    "logistic_regression": {"max_iter": 2000},
    "mlp": {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True},
}
NEG_SET_LABELS = {"n0": "N0 (original)", "n2": "N2 (degree-matched)"}


def build_topology_matrix(pairs_df: pd.DataFrame, degree: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    da = pairs_df["protein_a"].map(degree).to_numpy(dtype=float)
    db = pairs_df["protein_b"].map(degree).to_numpy(dtype=float)
    x = np.column_stack([da + db, np.abs(da - db), np.minimum(da, db), np.maximum(da, db)])
    return x, pairs_df["label"].to_numpy()


def load_handcrafted():
    feat_dir = REPO_ROOT / "data" / "processed" / "multi_species" / "features"
    aac = pd.read_csv(feat_dir / "aac.csv")
    ctd = pd.read_csv(feat_dir / "ctd.csv")
    return aac.merge(ctd, on="protein_id")


def main():
    set_seed(SEED)
    ms_dir = REPO_ROOT / "data" / "processed" / "multi_species"
    pairs = pd.read_csv(ms_dir / "ppi_pairs_clean.tsv", sep="\t", header=None, names=["protein_a", "protein_b", "label"])
    proteins = pd.read_csv(ms_dir / "proteins_with_groups.tsv", sep="\t")
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=10)
    degree = protein_stats["positive_degree"]
    known_pairs = _known_pair_set(pos)

    logger.info(f"Positives: {len(pos)}, N0 negatives: {len(neg0)}, proteins: {len(proteins)}")
    datasets = build_r0_n0_n2_n5_datasets(pos, neg0, protein_stats, known_pairs, SEED, TEST_FRACTION)
    n0_train, n0_test = datasets["n0"]
    n2_train, n2_test = datasets["n2"]

    # --- Degree-balance diagnostic (same SMD/KS/Wasserstein suite as the Guo analysis) ---
    balance_rows = []
    for neg_name, (train_df, test_df) in [("n0", (n0_train, n0_test)), ("n2", (n2_train, n2_test))]:
        neg_all = pd.concat([train_df[train_df["label"] == 0], test_df[test_df["label"] == 0]], ignore_index=True)
        c = compare_degree_distributions(pos, neg_all, degree, N_BOOT, SEED)
        logger.info(f"[positive vs {neg_name}] mean_diff={c['mean_diff']:.2f} SMD={c['standardized_mean_difference']:.3f} "
                    f"(well_balanced<0.1: {c['smd_well_balanced']}) KS={c['ks_statistic']:.3f} Wasserstein={c['wasserstein_distance']:.2f}")
        balance_rows.append({"comparison": f"positive_vs_{neg_name}", **c})
    balance_df = pd.DataFrame(balance_rows)
    tables_dir = REPO_ROOT / "tables"
    balance_df.to_csv(tables_dir / "multi_species_degree_balance.csv", index=False)
    logger.info(f"Wrote {tables_dir}/multi_species_degree_balance.csv")

    # --- Representations: handcrafted always; ESM-2 if scripts/41 has finished ---
    representations = {"aac_ctd": load_handcrafted()}
    esm2_path = REPO_ROOT / "embeddings" / "esm2_multi_species" / "protein_embeddings.h5"
    if esm2_path.exists():
        representations["esm2_mean"] = load_esm2_features("mean", esm2_path.parent)
        logger.info("ESM-2 embeddings found -- including esm2_mean representation.")
    else:
        logger.warning("ESM-2 embeddings not found yet (scripts/41) -- reporting handcrafted (AAC+CTD) results only.")

    rows = []
    for neg_name, (train_df, test_df) in [("n0", (n0_train, n0_test)), ("n2", (n2_train, n2_test))]:
        # Degree-only baseline (no sequence information whatsoever)
        x_train, y_train = build_topology_matrix(train_df, degree)
        x_test, y_test = build_topology_matrix(test_df, degree)
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        model = build_model("logistic_regression", {}, SEED)
        model.fit(x_train_s, y_train)
        m = classification_metrics(y_test, model.predict_proba(x_test_s)[:, 1])
        logger.info(f"[{NEG_SET_LABELS[neg_name]} x degree_only_lr] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
        rows.append({"negative_set": neg_name, "negative_set_label": NEG_SET_LABELS[neg_name],
                     "representation": "degree_only", "model": "logistic_regression",
                     "n_train": len(train_df), "n_test": len(test_df), **m})

        for repr_name, protein_features in representations.items():
            x_train, y_train = build_pair_matrix(train_df, protein_features, fusion=FUSION)
            x_test, y_test = build_pair_matrix(test_df, protein_features, fusion=FUSION)
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)
            for model_name, params in SEQUENCE_MODELS.items():
                model = build_model(model_name, params, SEED)
                model.fit(x_train_s, y_train)
                m = classification_metrics(y_test, model.predict_proba(x_test_s)[:, 1])
                logger.info(f"[{NEG_SET_LABELS[neg_name]} x {repr_name} x {model_name}] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
                rows.append({"negative_set": neg_name, "negative_set_label": NEG_SET_LABELS[neg_name],
                             "representation": repr_name, "model": model_name,
                             "n_train": len(train_df), "n_test": len(test_df), **m})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(tables_dir / "multi_species_negative_sampling.csv", index=False)
    logger.info(f"Wrote {tables_dir}/multi_species_negative_sampling.csv")

    results_dir = REPO_ROOT / "results" / "multi_species_pooled"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "note": (
                "Compact independent-benchmark replication on the combined "
                "C. elegans/Drosophila/E. coli PPI set. R0 (random split) N0-vs-N2 only, reduced model "
                "set (degree-only LR, sequence LR, sequence MLP)."
            ),
            "representations_included": list(representations.keys()),
            "esm2_available": esm2_path.exists(),
            "results": rows,
        },
        results_dir / "model_comparison.json",
    )
    write_manifest(results_dir, config={"representations": list(representations.keys()), "test_fraction": TEST_FRACTION}, seed=SEED,
                    extra={"script": "42_multi_species_model_comparison.py"})
    logger.info("Multi-species model comparison complete.")


if __name__ == "__main__":
    main()
