"""Two-sided negative-construction check.

This addresses the most important methodological
objection to Section 3.5's negative-sampling confound finding: N2/N3/N5
are built by ONE-SIDED substitution (`sample_matched_negatives`) -- for
each real positive pair (u,v), one endpoint is kept fixed at a genuine
interacting protein and only the other is substituted with a decoy. This
means the N0-vs-N2 comparison conflates two changes at once, not one:

  N0 = the benchmark's own (undocumented) negative-generation process
  N2 = one-sided, positive-anchored substitution + degree matching

so the observed collapse could in principle come from the substitution
MECHANISM itself (every N2 decoy shares a real interactor with a genuine
positive pair; N0's own negatives are not known to have this property),
not from degree matching specifically.

This script adds N6: a TWO-SIDED negative construction
(`sample_two_sided_matched_negatives`) that draws BOTH decoy endpoints
fresh from the protein pool, matching the joint degree-decile distribution
of each source positive pair without requiring either endpoint to be a
real interactor. If N6 collapses performance the same way N2 does, that is
direct evidence the one-sided-substitution mechanism is NOT what drives
the confound -- degree matching alone reproduces it regardless of
mechanism. If N6 does NOT collapse as much as N2, that is an equally
important, honestly-reported finding that the substitution mechanism
itself contributes independently of degree.

Reuses the negative-sampling study's exact protocol (scripts/11_negative_sampling_study.py):
same seed, same test_fraction, same ESM-2+MLP config
(configs/hard_negative.yaml), same positive train/test split -- so N0/N2
results here are directly comparable to Table 7, and N6 is evaluated
alongside them under an identical protocol. Also reruns the topology-only
(degree-only) baseline (scripts/24_topology_only_baseline.py) on N0/N2/N6
as a second, cheaper confirmation.

Writes:
  data/processed/negative_sets/n6.tsv
  tables/two_sided_negative_check.csv
  figures/two_sided_negative_check.pdf
  results/two_sided_negative_check/
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
    sample_matched_negatives,
    sample_two_sided_matched_negatives,
)
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("two_sided_negative_check")

NEGATIVE_SET_LABELS = {
    "n0": "N0 (benchmark original)",
    "n2": "N2 (degree-matched, one-sided)",
    "n6": "N6 (degree-matched, two-sided)",
}


def build_topology_matrix(pairs_df: pd.DataFrame, scalar: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    da = pairs_df["protein_a"].map(scalar).to_numpy(dtype=float)
    db = pairs_df["protein_b"].map(scalar).to_numpy(dtype=float)
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
    known_pairs = _known_pair_set(pos)

    # --- N6: two-sided degree-matched, reusing N2's existing saved set for the one-sided comparison ---
    neg_sets_dir = REPO_ROOT / "data" / "processed" / "negative_sets"
    n2 = pd.read_csv(neg_sets_dir / "n2.tsv", sep="\t")
    n6 = sample_two_sided_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], seed)
    n6.to_csv(neg_sets_dir / "n6.tsv", sep="\t", index=False)
    overlap = set(zip(n6["protein_a"], n6["protein_b"])) & known_pairs
    assert not overlap, f"N6 contains {len(overlap)} pairs that are actually known positives"
    logger.info(f"N6 (two-sided degree-matched): {len(n6)} pairs (N2 one-sided: {len(n2)} pairs)")

    negative_sets = {"n0": neg0, "n2": n2, "n6": n6}

    # --- Shared positive train/test split, identical to script 11's (same seed/test_fraction) ---
    pos_train, pos_test = train_test_split(pos, test_size=config["test_fraction"], random_state=seed)
    train_sets, test_sets = {}, {}
    for name, neg_df in negative_sets.items():
        neg_train, neg_test = train_test_split(neg_df, test_size=config["test_fraction"], random_state=seed)
        train_sets[name] = pd.concat([pos_train, neg_train], ignore_index=True)
        test_sets[name] = pd.concat([pos_test, neg_test], ignore_index=True)

    rows = []

    # --- Sequence-based model (ESM-2 + MLP, identical config to Table 7): train on N0, evaluate on N0/N2/N6 ---
    protein_features = load_esm2_features(config["representation"].replace("esm2_", ""), REPO_ROOT / "embeddings" / "esm2")

    def fit_seq(train_df):
        x_train, y_train = build_pair_matrix(train_df, protein_features, fusion=config["fusion"])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = build_model(config["model"], config["model_params"], seed)
        model.fit(x_train, y_train)
        return model, scaler

    def eval_seq(model, scaler, test_df):
        x_test, y_test = build_pair_matrix(test_df, protein_features, fusion=config["fusion"])
        x_test = scaler.transform(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        return classification_metrics(y_test, y_prob)

    logger.info("=== ESM-2+MLP: train on N0, evaluate on N0/N2/N6 test sets ===")
    seq_model, seq_scaler = fit_seq(train_sets["n0"])
    for name in negative_sets:
        m = eval_seq(seq_model, seq_scaler, test_sets[name])
        logger.info(f"[ESM-2+MLP: trained N0 -> tested {name}] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
        rows.append({"representation": "esm2_mlp", "trained_on": "n0", "evaluated_on": name, **m})

    # --- Degree-only topology baseline (Logistic Regression), same three sets ---
    scalar = protein_stats["positive_degree"]

    def fit_topo(train_df):
        x_train, y_train = build_topology_matrix(train_df, scalar)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = build_model("logistic_regression", {}, seed)
        model.fit(x_train, y_train)
        return model, scaler

    def eval_topo(model, scaler, test_df):
        x_test, y_test = build_topology_matrix(test_df, scalar)
        x_test = scaler.transform(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        return classification_metrics(y_test, y_prob)

    logger.info("=== Degree-only LR: train on N0, evaluate on N0/N2/N6 test sets ===")
    topo_model, topo_scaler = fit_topo(train_sets["n0"])
    for name in negative_sets:
        m = eval_topo(topo_model, topo_scaler, test_sets[name])
        logger.info(f"[Degree-only LR: trained N0 -> tested {name}] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
        rows.append({"representation": "degree_only_lr", "trained_on": "n0", "evaluated_on": name, **m})

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    results_df.to_csv(tables_dir / "two_sided_negative_check.csv", index=False)
    logger.info(f"Wrote {tables_dir}/two_sided_negative_check.csv")

    # --- Figure: N0 vs N2 (one-sided) vs N6 (two-sided), both representations ---
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["n0", "n2", "n6"]
    labels = [NEGATIVE_SET_LABELS[n] for n in order]
    x = np.arange(len(order))
    width = 0.35
    seq_vals = [results_df[(results_df["representation"] == "esm2_mlp") & (results_df["evaluated_on"] == n)]["roc_auc"].iloc[0] for n in order]
    topo_vals = [results_df[(results_df["representation"] == "degree_only_lr") & (results_df["evaluated_on"] == n)]["roc_auc"].iloc[0] for n in order]
    ax.bar(x - width / 2, seq_vals, width, label="ESM-2 + MLP", color="#d95f02")
    ax.bar(x + width / 2, topo_vals, width, label="Degree-only (LR)", color="#1b9e77")
    ax.axhline(0.5, color="gray", linestyle="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Trained on N0, evaluated on N0 / N2 (one-sided) / N6 (two-sided)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    figures_dir = REPO_ROOT / "figures"
    fig.savefig(figures_dir / "two_sided_negative_check.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/two_sided_negative_check.pdf")

    results_dir = REPO_ROOT / "results" / "two_sided_negative_check"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "n2_size": len(n2), "n6_size": len(n6),
            "note": (
                "N6 draws both decoy endpoints fresh from the degree-decile-matched protein pool "
                "(sample_two_sided_matched_negatives), unlike N2's one-sided substitution "
                "(sample_matched_negatives), which always anchors one decoy endpoint at a real "
                "positive's own protein. Comparable N2-vs-N6 collapse isolates degree matching "
                "from the substitution mechanism itself."
            ),
        },
        results_dir / "two_sided_check_summary.json",
    )
    write_manifest(results_dir, config=config, seed=seed, extra={"script": "26_two_sided_negative_check.py"})
    logger.info("Two-sided negative-construction check complete.")


if __name__ == "__main__":
    main()
