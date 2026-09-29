"""Does N2's decile-matching granularity
explain why the multi-species replication's sequence models (AAC+CTD,
ESM-2) retain ROC-AUC ~0.60-0.70 on N2 (scripts/42), instead of collapsing
toward chance the way they do on the yeast benchmark's N2 (Section 3.6)?

scripts/42's degree-balance diagnostic already flagged a plausible reason:
this dataset's positive-vs-N2 SMD is 0.130, above the 0.1 well-balanced
threshold used throughout the rest of this study -- unlike the yeast
benchmark's N2, where SMD=0.011 is two orders of magnitude under it
(Section 2.6). The yeast benchmark already tested whether decile coarseness itself explains a similar gap
there and found it does not (N2-decile/N2-exact/N2-nn are statistically
indistinguishable, all ~chance). This script runs the identical
N2-decile / N2-exact / N2-nn comparison on the multi-species dataset, using
scripts/42's exact protocol (same R0 split machinery, same reduced model
set, same "consistent" train/test negative-set design) instead of script
33's degree-only Experiment-A design, so it also exercises the sequence
models the residual signal actually shows up in -- not just the
degree-only baseline.

Writes:
  tables/multi_species_degree_matching_sensitivity.csv
  tables/multi_species_degree_matching_sensitivity_balance.csv
  results/multi_species_pooled/degree_matching_sensitivity.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.negative_sampling import (
    _known_pair_set,
    compute_protein_stats,
    sample_exact_degree_matched_negatives,
    sample_matched_negatives,
    sample_nearest_degree_matched_negatives,
)
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
logger = get_logger("multi_species_degree_matching_sensitivity")

SEED = 42
TEST_FRACTION = 0.20
N_BOOT = 2000
FUSION = "combined"
SEQUENCE_MODELS = {
    "logistic_regression": {"max_iter": 2000},
    "mlp": {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True},
}
VARIANT_LABELS = {
    "n2_decile": "N2 (decile-matched, scripts/42)",
    "n2_exact": "N2-exact (exact degree)",
    "n2_nn": "N2-nn (nearest-neighbor degree)",
}


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


def decoy_reuse_stats(neg_df: pd.DataFrame) -> dict:
    """Per-protein appearance count across a negative set (either endpoint) --
    diagnoses the decoy-overconcentration artifact fixed in
    sample_nearest_degree_matched_negatives (src/data/negative_sampling.py)."""
    counts = pd.concat([neg_df["protein_a"], neg_df["protein_b"]]).value_counts()
    return {
        "n_unique_decoy_proteins": int(counts.shape[0]),
        "max_reuse_count": int(counts.max()),
        "max_reuse_fraction": float(counts.max() / len(neg_df)),
        "median_reuse_count": float(counts.median()),
        "p95_reuse_count": float(counts.quantile(0.95)),
    }


def main():
    set_seed(SEED)
    ms_dir = REPO_ROOT / "data" / "processed" / "multi_species"
    pairs = pd.read_csv(ms_dir / "ppi_pairs_clean.tsv", sep="\t", header=None, names=["protein_a", "protein_b", "label"])
    proteins = pd.read_csv(ms_dir / "proteins_with_groups.tsv", sep="\t")
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=10)
    degree = protein_stats["positive_degree"]
    known_pairs = _known_pair_set(pos)

    n2_decile = sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], SEED)
    n2_exact = sample_exact_degree_matched_negatives(pos, protein_stats, known_pairs, SEED)
    n2_nn = sample_nearest_degree_matched_negatives(pos, protein_stats, known_pairs, SEED)
    logger.info(f"N2-decile: {len(n2_decile)} pairs")
    logger.info(f"N2-exact: {len(n2_exact)} pairs ({n2_exact.attrs['n_skipped_no_exact_match']} positive pairs skipped, no exact-degree decoy)")
    logger.info(f"N2-nn: {len(n2_nn)} pairs ({n2_nn.attrs['n_skipped_no_neighbor']} positive pairs skipped)")
    negative_variants = {"n2_decile": n2_decile, "n2_exact": n2_exact, "n2_nn": n2_nn}

    pos_train, pos_test = train_test_split(pos, test_size=TEST_FRACTION, random_state=SEED)
    datasets = {}
    for name, neg_df in negative_variants.items():
        neg_train, neg_test = train_test_split(neg_df, test_size=TEST_FRACTION, random_state=SEED)
        datasets[name] = (
            pd.concat([pos_train, neg_train], ignore_index=True),
            pd.concat([pos_test, neg_test], ignore_index=True),
        )

    # --- Degree-balance diagnostic for each matching granularity ---
    balance_rows = []
    for name, neg_df in negative_variants.items():
        c = compare_degree_distributions(pos, neg_df, degree, N_BOOT, SEED)
        reuse = decoy_reuse_stats(neg_df)
        logger.info(f"[positive vs {name}] mean_diff={c['mean_diff']:.2f} SMD={c['standardized_mean_difference']:.3f} "
                    f"(well_balanced<0.1: {c['smd_well_balanced']}) KS={c['ks_statistic']:.3f} Wasserstein={c['wasserstein_distance']:.2f}")
        logger.info(f"[{name} decoy reuse] n_unique={reuse['n_unique_decoy_proteins']} max={reuse['max_reuse_count']} "
                    f"({reuse['max_reuse_fraction']:.2%}) median={reuse['median_reuse_count']:.1f} p95={reuse['p95_reuse_count']:.1f}")
        balance_rows.append({"variant": name, "variant_label": VARIANT_LABELS[name], **c, **reuse})
    balance_df = pd.DataFrame(balance_rows)
    tables_dir = REPO_ROOT / "tables"
    balance_df.to_csv(tables_dir / "multi_species_degree_matching_sensitivity_balance.csv", index=False)
    logger.info(f"Wrote {tables_dir}/multi_species_degree_matching_sensitivity_balance.csv")

    # --- Representations: handcrafted + ESM-2 (both available at this point) ---
    representations = {"aac_ctd": load_handcrafted()}
    esm2_path = REPO_ROOT / "embeddings" / "esm2_multi_species" / "protein_embeddings.h5"
    representations["esm2_mean"] = load_esm2_features("mean", esm2_path.parent)

    rows = []
    for name, (train_df, test_df) in datasets.items():
        x_train, y_train = build_topology_matrix(train_df, degree)
        x_test, y_test = build_topology_matrix(test_df, degree)
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        model = build_model("logistic_regression", {}, SEED)
        model.fit(x_train_s, y_train)
        m = classification_metrics(y_test, model.predict_proba(x_test_s)[:, 1])
        logger.info(f"[{VARIANT_LABELS[name]} x degree_only_lr] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
        rows.append({"variant": name, "variant_label": VARIANT_LABELS[name],
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
                logger.info(f"[{VARIANT_LABELS[name]} x {repr_name} x {model_name}] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
                rows.append({"variant": name, "variant_label": VARIANT_LABELS[name],
                             "representation": repr_name, "model": model_name,
                             "n_train": len(train_df), "n_test": len(test_df), **m})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(tables_dir / "multi_species_degree_matching_sensitivity.csv", index=False)
    logger.info(f"Wrote {tables_dir}/multi_species_degree_matching_sensitivity.csv")

    results_dir = REPO_ROOT / "results" / "multi_species_pooled"
    save_json(
        {
            "note": (
                "Tests whether N2's decile-matching "
                "granularity (Section 2.6) explains why the multi-species replication's "
                "sequence models retain ROC-AUC ~0.60-0.70 on N2 (scripts/42) rather than "
                "collapsing toward chance as on the yeast benchmark's N2. Compares decile, "
                "exact-degree, and nearest-neighbor degree matching using scripts/42's exact "
                "model set and 'consistent' train/test protocol."
            ),
            "negative_set_sizes": {k: len(v) for k, v in negative_variants.items()},
            "n2_exact_skipped_pairs": int(n2_exact.attrs["n_skipped_no_exact_match"]),
            "n2_nn_skipped_pairs": int(n2_nn.attrs["n_skipped_no_neighbor"]),
            "results": rows,
        },
        results_dir / "degree_matching_sensitivity.json",
    )
    write_manifest(results_dir, config={"variants": list(negative_variants.keys())}, seed=SEED,
                    extra={"script": "43_multi_species_degree_matching_sensitivity.py"})
    logger.info("Multi-species degree-matching granularity sensitivity complete.")


if __name__ == "__main__":
    main()
