"""Topology-only baseline: directly tests whether network degree alone
explains the negative-sampling confound (Section 3.5), as opposed to some
other property that happens to covary with N2's degree-matched construction.

Reuses the negative-sampling study's exact protocol (scripts/11_negative_sampling_study.py) --
same seed (42), same test_fraction (0.20), same saved N1/N2/N3/N5 negative
sets on disk -- so results are directly comparable to Table 7, with the
ESM-2+MLP pair representation replaced by four symmetric functions of the
two proteins' positive-degree alone (sum, |difference|, min, max) and no
sequence information whatsoever. If this collapses on N2 the same way the
sequence-based models do (Table 7), that is direct evidence degree itself
-- not an artifact of N2's construction process -- drives the confound.

Two models are tried on the topology-only features: Logistic Regression
(simplest, most interpretable -- can degree alone be linearly separated?)
and Random Forest (checks for a non-linear degree effect a linear model
would miss).

Writes:
  tables/topology_only_baseline.csv
  figures/topology_only_baseline.pdf
  results/topology_baseline/
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
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("topology_only_baseline")

NEGATIVE_SET_LABELS = {
    "n0": "N0 (benchmark original)",
    "n1": "N1 (random unknown)",
    "n2": "N2 (degree-matched)",
    "n3": "N3 (length-matched)",
    "n5": "N5 (hard: degree+length)",
}
MODELS = {
    "logistic_regression": ("logistic_regression", {}),
    "random_forest": ("random_forest", {"n_estimators": 200, "max_depth": 6}),
}


def build_topology_matrix(pairs_df: pd.DataFrame, scalar: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Four symmetric functions of a single per-protein scalar (positive
    degree, or -- as a specificity check -- sequence length) for the two
    endpoints: sum, |diff|, min, max -- so f(a,b) = f(b,a) by construction
    (Section 2.2's fusion philosophy applied to a scalar instead of an
    embedding). No sequence content (residues/composition) is used either
    way; only degree carries network-topology information."""
    da = pairs_df["protein_a"].map(scalar).to_numpy(dtype=float)
    db = pairs_df["protein_b"].map(scalar).to_numpy(dtype=float)
    x = np.column_stack([da + db, np.abs(da - db), np.minimum(da, db), np.maximum(da, db)])
    y = pairs_df["label"].to_numpy()
    return x, y


def plot_results(results_df):
    """Refresh the comparison figure from saved metrics without fitting models."""
    # --- Comparison figure: topology-only (LR) vs. the full ESM-2+MLP model (Table 7) ---
    table7 = pd.read_csv(REPO_ROOT / "tables" / "table7_negative_sampling_robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.35

    exp_a_topo = results_df[(results_df["experiment"] == "A_train_n0_eval_swap") & (results_df["model"] == "logistic_regression") & (results_df["feature_source"] == "degree")]
    exp_a_full = table7[table7["experiment"] == "A_train_n0_eval_swap"]
    order = ["n0", "n1", "n3", "n5", "n2"]
    labels = [NEGATIVE_SET_LABELS[n].split(" (")[0] for n in order]
    x = np.arange(len(order))
    topo_vals = [exp_a_topo.set_index("evaluated_on").loc[n, "roc_auc"] for n in order]
    full_vals = [exp_a_full.set_index("evaluated_on").loc[n, "roc_auc"] for n in order]
    axes[0].bar(x - width / 2, full_vals, width, label="ESM-2 + MLP", color="#d95f02")
    axes[0].bar(x + width / 2, topo_vals, width, label="Topology-only (degree, LR)", color="#1b9e77")
    axes[0].axhline(0.5, color="gray", linestyle="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Exp A: trained on N0,\nevaluated on swapped test negatives")
    axes[0].legend(fontsize=8)

    exp_b_topo = results_df[(results_df["experiment"] == "B_train_each_eval_n0") & (results_df["model"] == "logistic_regression") & (results_df["feature_source"] == "degree")]
    exp_b_full = table7[table7["experiment"] == "B_train_each_eval_n0"]
    topo_vals_b = [exp_b_topo.set_index("trained_on").loc[n, "roc_auc"] for n in order]
    full_vals_b = [exp_b_full.set_index("trained_on").loc[n, "roc_auc"] for n in order]
    axes[1].bar(x - width / 2, full_vals_b, width, label="ESM-2 + MLP", color="#d95f02")
    axes[1].bar(x + width / 2, topo_vals_b, width, label="Topology-only (degree, LR)", color="#1b9e77")
    axes[1].axhline(0.5, color="gray", linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Exp B: trained on each negative set,\nevaluated on N0 (common benchmark)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Topology-only (degree) baseline vs. full sequence-based model")
    fig.tight_layout()
    figures_dir = REPO_ROOT / "figures"
    fig.savefig(figures_dir / "topology_only_baseline.pdf")
    fig.savefig(figures_dir / "topology_only_baseline.png", dpi=300)
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/topology_only_baseline.pdf")


def main():
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    config = load_config(REPO_ROOT / "configs" / "hard_negative.yaml")
    seed = config["seed"]
    set_seed(seed)

    # --- Exact reload of the negative-sampling study's inputs, so splits are identical to Table 7 ---
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=config["n_deciles"])
    feature_sources = {"degree": protein_stats["positive_degree"], "length": protein_stats["length"]}

    neg_sets_dir = REPO_ROOT / "data" / "processed" / "negative_sets"
    negative_sets = {
        "n0": neg0,
        "n1": pd.read_csv(neg_sets_dir / "n1.tsv", sep="\t"),
        "n2": pd.read_csv(neg_sets_dir / "n2.tsv", sep="\t"),
        "n3": pd.read_csv(neg_sets_dir / "n3.tsv", sep="\t"),
        "n5": pd.read_csv(neg_sets_dir / "n5.tsv", sep="\t"),
    }

    pos_train, pos_test = train_test_split(pos, test_size=config["test_fraction"], random_state=seed)
    train_sets, test_sets = {}, {}
    for name, neg_df in negative_sets.items():
        neg_train, neg_test = train_test_split(neg_df, test_size=config["test_fraction"], random_state=seed)
        train_sets[name] = pd.concat([pos_train, neg_train], ignore_index=True)
        test_sets[name] = pd.concat([pos_test, neg_test], ignore_index=True)

    def fit(model_key, scalar, train_df):
        model_name, extra_params = MODELS[model_key]
        x_train, y_train = build_topology_matrix(train_df, scalar)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = build_model(model_name, extra_params, seed)
        model.fit(x_train, y_train)
        return model, scaler

    def evaluate(model, scaler, scalar, test_df):
        x_test, y_test = build_topology_matrix(test_df, scalar)
        x_test = scaler.transform(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        return classification_metrics(y_test, y_prob)

    rows = []
    for feature_name, scalar in feature_sources.items():
        for model_key in MODELS:
            logger.info(f"=== {feature_name}-only baseline: {model_key} ===")
            # Experiment A: train once on N0, evaluate on swapped test negatives.
            model_n0, scaler_n0 = fit(model_key, scalar, train_sets["n0"])
            for name in negative_sets:
                m = evaluate(model_n0, scaler_n0, scalar, test_sets[name])
                logger.info(f"[A: trained N0 -> tested {name}, {feature_name}/{model_key}] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
                rows.append({"feature_source": feature_name, "model": model_key, "experiment": "A_train_n0_eval_swap", "trained_on": "n0", "evaluated_on": name, **m})

            # Experiment B: train separately on each set, evaluate all on N0's test set.
            for name in negative_sets:
                model, scaler = fit(model_key, scalar, train_sets[name])
                m = evaluate(model, scaler, scalar, test_sets["n0"])
                logger.info(f"[B: trained {name} -> tested N0, {feature_name}/{model_key}] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
                rows.append({"feature_source": feature_name, "model": model_key, "experiment": "B_train_each_eval_n0", "trained_on": name, "evaluated_on": "n0", **m})

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "topology_only_baseline.csv", index=False)
    logger.info(f"Wrote {tables_dir}/topology_only_baseline.csv")

    plot_results(results_df)

    negative_sampling_study_topo_dir = REPO_ROOT / "results" / "topology_baseline"
    negative_sampling_study_topo_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "features": ["<scalar>_sum", "<scalar>_abs_diff", "<scalar>_min", "<scalar>_max"],
            "feature_sources": {
                "degree": "positive_degree from compute_protein_stats (same degree used to construct N2/N5 decoys)",
                "length": "sequence length from compute_protein_stats -- specificity check: is any single scalar protein property this separable, or is it degree specifically?",
            },
            "note": (
                "No sequence content (residues/composition) used either way. Reuses the negative-sampling study's "
                "exact seed/splits/negative sets (scripts/11_negative_sampling_study.py) for "
                "direct comparability with Table 7."
            ),
        },
        negative_sampling_study_topo_dir / "topology_baseline_summary.json",
    )
    write_manifest(negative_sampling_study_topo_dir, config=config, seed=seed, extra={"script": "24_topology_only_baseline.py"})
    logger.info("Topology-only baseline complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Render saved metrics without training models")
    if parser.parse_args().plot_only:
        plot_results(pd.read_csv(REPO_ROOT / "tables" / "topology_only_baseline.csv"))
    else:
        main()
