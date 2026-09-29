"""Multi-species external-benchmark audit + preprocessing.

First step of the compact replication (see configs/multi_species.yaml):
verify the dataset's integrity the same way the dataset audit (scripts/01) did for
the Guo yeast benchmark, canonicalize pairs the same way the canonicalization
(scripts/02) did, and report the same dataset-dimension/degree-distribution
facts the main study's headline confound diagnostic depends on -- so the
replication is directly comparable, not just superficially similar.

Writes:
  data/processed/multi_species/ppi_pairs_clean.tsv
  data/processed/multi_species/proteins_with_groups.tsv
  results/multi_species_pooled/audit.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import validate_data
from src.data.canonicalize import canonicalize_pairs
from src.data.load_data import load_original_split, load_pairs, load_proteins
from src.data.sequence_groups import assign_sequence_groups
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("multi_species_audit")


def main():
    config = load_config(REPO_ROOT / "configs" / "multi_species.yaml")
    proteins = load_proteins(config, REPO_ROOT)
    pairs = load_pairs(config, REPO_ROOT)
    train_split, test_split = load_original_split(config, REPO_ROOT)

    report = {
        "dimensions": {
            "n_proteins": int(proteins["protein_id"].nunique()),
            "n_pairs": int(len(pairs)),
            "n_positive": int((pairs["label"] == 1).sum()),
            "n_negative": int((pairs["label"] == 0).sum()),
        },
        "integrity": {
            "missing_sequences": len(validate_data.missing_sequences(proteins)),
            "duplicate_protein_ids": len(validate_data.duplicate_protein_ids(proteins)),
            "duplicated_pairs_exact_rows": len(validate_data.duplicated_pairs_exact(pairs)),
            "reverse_duplicate_pair_rows": len(validate_data.reverse_duplicate_pairs(pairs)),
            "conflicting_label_rows": len(validate_data.conflicting_labels(pairs)),
            "self_interactions": len(validate_data.self_interactions(pairs)),
            "pairs_referencing_unknown_proteins": len(validate_data.pairs_referencing_unknown_proteins(pairs, proteins)),
        },
    }
    logger.info(f"Dimensions: {report['dimensions']}")
    logger.info(f"Integrity: {report['integrity']}")

    train_proteins = set(train_split["protein_a"]) | set(train_split["protein_b"])
    test_proteins = set(test_split["protein_a"]) | set(test_split["protein_b"])
    overlap = train_proteins & test_proteins
    report["original_split_audit"] = {
        "train_pairs": int(len(train_split)), "test_pairs": int(len(test_split)),
        "train_unique_proteins": len(train_proteins), "test_unique_proteins": len(test_proteins),
        "protein_overlap_count": len(overlap),
        "protein_overlap_fraction_of_test": len(overlap) / len(test_proteins) if test_proteins else 0.0,
        "note": "Naive pre-supplied split, protein overlap expected -- same status as the yeast benchmark's train_cmap/test_cmap split (Section 2.1).",
    }
    logger.info(f"Original split protein overlap: {len(overlap)}/{len(test_proteins)} test proteins also in train")

    # --- Canonicalize pairs ---
    clean_pairs = canonicalize_pairs(pairs)
    n_dropped = clean_pairs.attrs["n_dropped_duplicates"]

    # Unlike the Guo yeast benchmark (0 self-interactions, the dataset audit), this
    # benchmark carries 1,279 self-interaction rows (protein_a == protein_b),
    # all labeled positive -- plausible homodimer annotations, not a label
    # conflict. N2's one-sided decoy-substitution logic (src/data/negative_
    # sampling.py) does not error on these, but a self-pair inflates its
    # protein's degree count by 2 for a single self-loop (counted once as
    # protein_a, once as protein_b) rather than the 1 a true self-loop
    # contributes -- a documented anomaly, corrected here by excluding
    # self-interactions before degree computation and N2 construction,
    # consistent with standard PPI-network analysis practice.
    n_self = int((clean_pairs["protein_a"] == clean_pairs["protein_b"]).sum())
    clean_pairs = clean_pairs[clean_pairs["protein_a"] != clean_pairs["protein_b"]].reset_index(drop=True)

    processed_dir = REPO_ROOT / "data" / "processed" / "multi_species"
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_pairs.to_csv(processed_dir / "ppi_pairs_clean.tsv", sep="\t", header=False, index=False)
    report["canonicalization"] = {
        "raw_pair_count": int(len(pairs)), "canonical_pair_count": int(len(clean_pairs)),
        "exact_duplicate_pairs_dropped": int(n_dropped),
        "self_interactions_dropped": n_self,
        "canonical_label_balance": {"positive": int((clean_pairs["label"] == 1).sum()), "negative": int((clean_pairs["label"] == 0).sum())},
    }
    logger.info(f"Canonicalized: {len(pairs)} raw -> {len(clean_pairs)} rows "
                f"({n_dropped} exact duplicates + {n_self} self-interactions dropped)")

    grouped = assign_sequence_groups(proteins)
    grouped.to_csv(processed_dir / "proteins_with_groups.tsv", sep="\t", index=False)

    # --- Degree distribution (the confound diagnostic this replication tests) ---
    pos = clean_pairs[clean_pairs["label"] == 1].reset_index(drop=True)
    neg0 = clean_pairs[clean_pairs["label"] == 0].reset_index(drop=True)
    import pandas as pd
    degree = pd.concat([pos["protein_a"], pos["protein_b"]]).value_counts()
    pos_deg_vals = pd.concat([pos["protein_a"].map(degree), pos["protein_b"].map(degree)]).fillna(0)
    neg0_deg_vals = pd.concat([neg0["protein_a"].map(degree), neg0["protein_b"].map(degree)]).fillna(0)
    report["degree_distribution"] = {
        "positive_mean_pair_associated_degree": float(pos_deg_vals.mean()),
        "n0_mean_pair_associated_degree": float(neg0_deg_vals.mean()),
        "mean_diff": float(pos_deg_vals.mean() - neg0_deg_vals.mean()),
        "n_proteins_zero_positive_degree": int((degree.reindex(proteins["protein_id"]).fillna(0) == 0).sum()),
    }
    logger.info(f"Degree distribution: positive mean={report['degree_distribution']['positive_mean_pair_associated_degree']:.2f}, "
                f"N0 mean={report['degree_distribution']['n0_mean_pair_associated_degree']:.2f}, "
                f"diff={report['degree_distribution']['mean_diff']:.2f}")

    results_dir = REPO_ROOT / "results" / "multi_species_pooled"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, results_dir / "audit.json")
    write_manifest(results_dir, config=config, seed=config.get("seed"), extra={"script": "39_multi_species_audit.py"})
    logger.info("Multi-species audit complete.")


if __name__ == "__main__":
    main()
