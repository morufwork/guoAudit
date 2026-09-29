"""Homology-controlled negative-sampling sensitivity analysis: N2/N3/N5's decile matching (Section 2.6) constrains only
degree and/or length -- explicitly not sequence homology (a disclosed scope
limit, Section 4.3). The concern: a degree-matched decoy could
happen to be a close sequence homolog, family member, or (by proxy, absent
complex/family annotation -- Section 2.6's N4 exclusion) functionally
related to the fixed protein it is paired against, which could affect
sequence-based model performance for reasons unrelated to degree matching
itself.

Constructs **N7**: the same one-sided degree+length decile-matched
substitution as N5, with an added exclusion -- a candidate decoy sharing the
fixed endpoint's MMseqs2 sequence cluster (Section 2.3's own 20%-identity
clustering, already computed for the homology-controlled splits, reused here
rather than re-run) is never accepted. N7 therefore matches on degree,
length, AND excludes detectable sequence homology to the fixed protein in
one construction -- everything except subcellular
localization, which (like N4) this dataset has no annotation to support
(Section 2.6).

Reuses the negative-sampling study's exact protocol (scripts/11_negative_sampling_study.py):
same seed, same ESM-2-mean + MLP model, same train/test fraction -- so N7 is
directly comparable to Table 7's N0/N2/N5.

Writes:
  data/processed/negative_sets/n7.tsv
  tables/homology_controlled_negative.csv
  figures/homology_controlled_negative.pdf
  results/cluster_excluded_negative/
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
    sample_homology_controlled_matched_negatives,
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
logger = get_logger("homology_controlled_negative")

VARIANT_LABELS = {
    "n0": "N0 (benchmark original)",
    "n2": "N2 (degree-matched)",
    "n5": "N5 (degree+length-matched)",
    "n7": "N7 (degree+length-matched, homology-excluded)",
}
HOMOLOGY_THRESHOLD = "20"  # reuses this study's own "detectable homolog" cutoff (Section 3.3, 4.3)


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

    cluster_tsv = REPO_ROOT / "data" / "processed" / "clusters" / f"cluster_{HOMOLOGY_THRESHOLD}.tsv"
    cluster_df = pd.read_csv(cluster_tsv, sep="\t")  # already-processed protein_id -> cluster_id (Section 2.3), not the raw mmseqs 2-column format
    cluster_assignment = dict(zip(cluster_df["protein_id"], cluster_df["cluster_id"]))
    n_clusters = cluster_df["cluster_id"].nunique()
    logger.info(f"Loaded {len(cluster_assignment)} protein->cluster assignments at {HOMOLOGY_THRESHOLD}% identity ({n_clusters} clusters)")

    neg_sets_dir = REPO_ROOT / "data" / "processed" / "negative_sets"
    n2 = pd.read_csv(neg_sets_dir / "n2.tsv", sep="\t")
    n5 = pd.read_csv(neg_sets_dir / "n5.tsv", sep="\t")
    n7 = sample_homology_controlled_matched_negatives(
        pos, protein_stats, known_pairs, ["degree_decile", "length_decile"], seed, cluster_assignment,
    )
    n7.to_csv(neg_sets_dir / "n7.tsv", sep="\t", index=False)

    # Sanity check: no N7 pair should share a cluster between its two endpoints.
    n7_same_cluster = sum(cluster_assignment.get(a) == cluster_assignment.get(b) for a, b in zip(n7["protein_a"], n7["protein_b"]))
    assert n7_same_cluster == 0, f"N7 contains {n7_same_cluster} pairs sharing a sequence cluster -- homology exclusion failed"

    logger.info(f"N5 (degree+length, no homology control): {len(n5)} pairs")
    logger.info(f"N7 (degree+length, homology-excluded): {len(n7)} pairs ({n7.attrs['n_skipped_homology_exhausted']} positive pairs skipped specifically because every remaining bucket candidate was a homolog of the fixed endpoint)")

    negative_sets = {"n0": neg0, "n2": n2, "n5": n5, "n7": n7}

    pos_train, pos_test = train_test_split(pos, test_size=config["test_fraction"], random_state=seed)
    train_sets, test_sets = {}, {}
    for name, neg_df in negative_sets.items():
        neg_train, neg_test = train_test_split(neg_df, test_size=config["test_fraction"], random_state=seed)
        train_sets[name] = pd.concat([pos_train, neg_train], ignore_index=True)
        test_sets[name] = pd.concat([pos_test, neg_test], ignore_index=True)

    protein_features = load_esm2_features(config["representation"].replace("esm2_", ""), REPO_ROOT / "embeddings" / "esm2")

    def fit(train_df):
        x_train, y_train = build_pair_matrix(train_df, protein_features, fusion=config["fusion"])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = build_model(config["model"], config["model_params"], seed)
        model.fit(x_train, y_train)
        return model, scaler

    def evaluate(model, scaler, test_df):
        x_test, y_test = build_pair_matrix(test_df, protein_features, fusion=config["fusion"])
        x_test = scaler.transform(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        return classification_metrics(y_test, y_prob)

    rows = []
    logger.info("=== Experiment A: train on N0 (ESM-2+MLP), evaluate on N0/N2/N5/N7 test sets ===")
    model_n0, scaler_n0 = fit(train_sets["n0"])
    for name in negative_sets:
        m = evaluate(model_n0, scaler_n0, test_sets[name])
        logger.info(f"[A: trained N0 -> tested {name}] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
        rows.append({"experiment": "A_train_n0_eval_swap", "trained_on": "n0", "evaluated_on": name, **m})

    logger.info("=== Experiment B: train on each set, evaluate on N0's common test set ===")
    for name in negative_sets:
        model, scaler = fit(train_sets[name])
        m = evaluate(model, scaler, test_sets["n0"])
        logger.info(f"[B: trained {name} -> tested N0] ROC-AUC={m['roc_auc']:.4f} MCC={m['mcc']:.4f}")
        rows.append({"experiment": "B_train_each_eval_n0", "trained_on": name, "evaluated_on": "n0", **m})

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "homology_controlled_negative.csv", index=False)
    logger.info(f"Wrote {tables_dir}/homology_controlled_negative.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    order = ["n0", "n2", "n5", "n7"]
    labels = [VARIANT_LABELS[n].split(" (")[0] for n in order]

    exp_a = results_df[results_df["experiment"] == "A_train_n0_eval_swap"].set_index("evaluated_on")
    axes[0].bar(labels, [exp_a.loc[n, "roc_auc"] for n in order], color="#1b9e77")
    axes[0].axhline(0.5, color="gray", linestyle="--")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Exp A: trained on N0,\nevaluated on swapped test negatives")

    exp_b = results_df[results_df["experiment"] == "B_train_each_eval_n0"].set_index("trained_on")
    axes[1].bar(labels, [exp_b.loc[n, "roc_auc"] for n in order], color="#d95f02")
    axes[1].axhline(0.5, color="gray", linestyle="--")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Exp B: trained on each set,\nevaluated on N0 (common benchmark)")

    fig.suptitle("Homology-controlled degree+length matching (N7) vs. N2/N5 (ESM-2 + MLP)")
    fig.tight_layout()
    figures_dir = REPO_ROOT / "figures"
    fig.savefig(figures_dir / "homology_controlled_negative.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/homology_controlled_negative.pdf")

    out_dir = REPO_ROOT / "results" / "cluster_excluded_negative"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "homology_threshold_pct": HOMOLOGY_THRESHOLD,
            "n_clusters_at_threshold": int(n_clusters),
            "negative_set_sizes": {k: len(v) for k, v in negative_sets.items()},
            "n7_skipped_homology_exhausted": int(n7.attrs["n_skipped_homology_exhausted"]),
            "n7_same_cluster_pairs_verified_zero": True,
            "note": (
                "N2/N3/N5 match only on degree and/or "
                "length (Section 2.6), never excluding sequence homology. N7 adds "
                "a homology exclusion (MMseqs2 clusters at 20% identity, Section "
                "2.3's own clustering, reused here) on top of N5's degree+length "
                "matching, so no N7 decoy shares a detectable-homology cluster "
                "with the protein it is paired against. Subcellular localization "
                "matching is not implemented for "
                "the same reason N4 was not constructed (Section 2.6): this "
                "dataset carries no GO/localization annotation."
            ),
        },
        out_dir / "homology_controlled_summary.json",
    )
    write_manifest(out_dir, config=config, seed=seed, extra={"script": "34_homology_controlled_negative.py"})
    logger.info("Homology-controlled negative-sampling sensitivity analysis complete.")


if __name__ == "__main__":
    main()
