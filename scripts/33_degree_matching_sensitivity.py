"""Degree-matching granularity sensitivity analysis. The concern is that N2's decile matching (Section 2.6) is
too coarse -- e.g. degree 1 and degree 3 can share a decile at the low-degree
end, or degree 10 and 15 at the high-degree end, since positive_degree is
discrete and heavily right-skewed (Section 3.1).

Compares three matching granularities for the same one-sided-substitution
construction (Section 2.6):
  N2       decile matching (already on disk, scripts/11_negative_sampling_study.py)
  N2-exact exact positive_degree matching, pairs skipped when no exact match exists
  N2-nn    true nearest-neighbor matching on raw positive_degree (no binning)

Propensity-score and kernel matching are not
implemented as separate variants here: with a single scalar confounder
(positive_degree), a propensity score is a monotonic function of degree
alone, so nearest-neighbor propensity-score matching selects the same decoys
as N2-nn's direct nearest-neighbor matching on degree. Kernel matching
produces a continuously-weighted synthetic control for outcome regression,
not a discrete decoy protein -- it has no natural realization in this
one-sided pair-substitution scheme, which requires a literal candidate
protein for a training/evaluation pair, so it is not applicable here (noted
as a scope limitation, not implemented as a fabricated approximation).

Reuses the topology-only baseline's exact protocol (scripts/24_topology_only_baseline.py):
same seed (42), same Experiment A design (train on N0, evaluate on swapped
test negatives), same degree-scalar features (sum/|diff|/min/max) and models
(Logistic Regression, Random Forest) -- so this directly re-tests Section
3.5's headline degree-only collapse (ROC-AUC 0.962/0.966 on N0, ~0.50 on N2)
under two strictly finer-grained matching schemes.

Writes:
  tables/degree_matching_sensitivity.csv
  figures/degree_matching_sensitivity.pdf
  results/degree_matching_sensitivity/
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

from src.data.negative_sampling import (
    _known_pair_set,
    compute_protein_stats,
    sample_exact_degree_matched_negatives,
    sample_nearest_degree_matched_negatives,
)
from src.evaluation.metrics import classification_metrics
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("degree_matching_sensitivity")

VARIANT_LABELS = {
    "n0": "N0 (benchmark original)",
    "n2_decile": "N2 (decile-matched)",
    "n2_exact": "N2-exact (exact degree)",
    "n2_nn": "N2-nn (nearest-neighbor degree)",
}
MODELS = {
    "logistic_regression": ("logistic_regression", {}),
    "random_forest": ("random_forest", {"n_estimators": 200, "max_depth": 6}),
}


def build_topology_matrix(pairs_df: pd.DataFrame, degree: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    da = pairs_df["protein_a"].map(degree).to_numpy(dtype=float)
    db = pairs_df["protein_b"].map(degree).to_numpy(dtype=float)
    x = np.column_stack([da + db, np.abs(da - db), np.minimum(da, db), np.maximum(da, db)])
    y = pairs_df["label"].to_numpy()
    return x, y


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
    known_pairs = _known_pair_set(pos)

    neg_sets_dir = REPO_ROOT / "data" / "processed" / "negative_sets"
    n2_decile = pd.read_csv(neg_sets_dir / "n2.tsv", sep="\t")
    n2_exact = sample_exact_degree_matched_negatives(pos, protein_stats, known_pairs, seed)
    n2_nn = sample_nearest_degree_matched_negatives(pos, protein_stats, known_pairs, seed)

    logger.info(f"N2 (decile): {len(n2_decile)} pairs")
    logger.info(f"N2-exact: {len(n2_exact)} pairs ({n2_exact.attrs['n_skipped_no_exact_match']} positive pairs skipped, no exact-degree decoy)")
    logger.info(f"N2-nn: {len(n2_nn)} pairs ({n2_nn.attrs['n_skipped_no_neighbor']} positive pairs skipped)")

    negative_sets = {"n0": neg0, "n2_decile": n2_decile, "n2_exact": n2_exact, "n2_nn": n2_nn}

    pos_train, pos_test = train_test_split(pos, test_size=config["test_fraction"], random_state=seed)
    train_sets, test_sets = {}, {}
    for name, neg_df in negative_sets.items():
        neg_train, neg_test = train_test_split(neg_df, test_size=config["test_fraction"], random_state=seed)
        train_sets[name] = pd.concat([pos_train, neg_train], ignore_index=True)
        test_sets[name] = pd.concat([pos_test, neg_test], ignore_index=True)

    def fit(model_key, train_df):
        model_name, extra_params = MODELS[model_key]
        x_train, y_train = build_topology_matrix(train_df, degree)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = build_model(model_name, extra_params, seed)
        model.fit(x_train, y_train)
        return model, scaler

    def evaluate(model, scaler, test_df):
        x_test, y_test = build_topology_matrix(test_df, degree)
        x_test = scaler.transform(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        return classification_metrics(y_test, y_prob)

    rows = []
    for model_key in MODELS:
        logger.info(f"=== degree-only baseline, matching-granularity sensitivity: {model_key} ===")
        model_n0, scaler_n0 = fit(model_key, train_sets["n0"])
        for name in negative_sets:
            m = evaluate(model_n0, scaler_n0, test_sets[name])
            logger.info(f"[trained N0 -> tested {name}, {model_key}] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
            rows.append({"model": model_key, "trained_on": "n0", "evaluated_on": name, "n_negatives": len(negative_sets[name]), **m})

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "degree_matching_sensitivity.csv", index=False)
    logger.info(f"Wrote {tables_dir}/degree_matching_sensitivity.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["n0", "n2_decile", "n2_exact", "n2_nn"]
    labels = [VARIANT_LABELS[n].split(" (")[0] for n in order]
    x = np.arange(len(order))
    width = 0.35
    lr = results_df[results_df["model"] == "logistic_regression"].set_index("evaluated_on")
    rf = results_df[results_df["model"] == "random_forest"].set_index("evaluated_on")
    ax.bar(x - width / 2, [lr.loc[n, "roc_auc"] for n in order], width, label="Logistic Regression", color="#1b9e77")
    ax.bar(x + width / 2, [rf.loc[n, "roc_auc"] for n in order], width, label="Random Forest", color="#d95f02")
    ax.axhline(0.5, color="gray", linestyle="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Degree-only baseline: trained on N0,\nevaluated across matching granularities")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures_dir = REPO_ROOT / "figures"
    fig.savefig(figures_dir / "degree_matching_sensitivity.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/degree_matching_sensitivity.pdf")

    out_dir = REPO_ROOT / "results" / "degree_matching_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "negative_set_sizes": {k: len(v) for k, v in negative_sets.items()},
            "n2_exact_skipped_pairs": int(n2_exact.attrs["n_skipped_no_exact_match"]),
            "n2_nn_skipped_pairs": int(n2_nn.attrs["n_skipped_no_neighbor"]),
            "note": (
                "Tests whether N2's decile-matching "
                "granularity (Section 2.6) drives the N0-vs-N2 collapse (Section "
                "3.5), by re-running the degree-only baseline's exact protocol "
                "(script 24) against exact-degree and nearest-neighbor degree "
                "matching in place of decile matching. Propensity-score and "
                "kernel matching are not implemented as separate variants: with "
                "one scalar confounder, propensity-score nearest-neighbor "
                "matching selects the same decoys as N2-nn, and kernel matching "
                "produces continuous synthetic controls with no discrete decoy "
                "protein to substitute into a real pair."
            ),
        },
        out_dir / "degree_matching_sensitivity_summary.json",
    )
    write_manifest(out_dir, config=config, seed=seed, extra={"script": "33_degree_matching_sensitivity.py"})
    logger.info("Degree-matching granularity sensitivity analysis complete.")


if __name__ == "__main__":
    main()
