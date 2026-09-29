"""Dataset audit.

Verifies the benchmark's stated dimensions, runs integrity checks, and
computes protein-level and network statistics. Writes:
  results/dataset_audit.json
  results/protein_statistics.csv
  results/pair_statistics.csv
  figures/class_distribution.pdf
  figures/sequence_length_distribution.pdf
  figures/network_degree_distribution.pdf

Anomalies found here must either be
corrected under a documented rule, or explicitly retained and justified —
this script does not silently fix anything, it only reports.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import audit, validate_data
from src.data.load_data import load_original_split, load_pairs, load_proteins
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("audit_dataset")


def main():
    config = load_config(REPO_ROOT / "configs" / "data.yaml")
    proteins = load_proteins(config, REPO_ROOT)
    pairs = load_pairs(config, REPO_ROOT)
    train_split, test_split = load_original_split(config, REPO_ROOT)

    report = {}

    # --- Dataset dimensions (verify, do not assume) ---
    dims = {
        "n_proteins": int(proteins["protein_id"].nunique()),
        "n_pairs": int(len(pairs)),
        "n_positive": int((pairs["label"] == 1).sum()),
        "n_negative": int((pairs["label"] == 0).sum()),
    }
    report["dimensions"] = dims
    report["dimensions_match_expected"] = dims == config["expected"]
    logger.info(f"Dimensions: {dims} (matches expected values: {report['dimensions_match_expected']})")

    # --- Integrity checks ---
    integrity = {
        "missing_sequences": len(validate_data.missing_sequences(proteins)),
        "missing_ids": len(validate_data.missing_ids(proteins)),
        "invalid_amino_acid_rows": len(validate_data.invalid_amino_acids(proteins, config["valid_amino_acids"])),
        "duplicate_protein_ids": len(validate_data.duplicate_protein_ids(proteins)),
        "identical_sequences_diff_ids_rows": len(validate_data.identical_sequences_diff_ids(proteins)),
        "duplicated_pairs_exact_rows": len(validate_data.duplicated_pairs_exact(pairs)),
        "reverse_duplicate_pair_rows": len(validate_data.reverse_duplicate_pairs(pairs)),
        "conflicting_label_rows": len(validate_data.conflicting_labels(pairs)),
        "self_interactions": len(validate_data.self_interactions(pairs)),
        "pairs_referencing_unknown_proteins": len(validate_data.pairs_referencing_unknown_proteins(pairs, proteins)),
    }
    report["integrity"] = integrity
    logger.info(f"Integrity: {integrity}")

    # --- Pre-supplied original split: protein-overlap check ---
    train_proteins = set(train_split["protein_a"]) | set(train_split["protein_b"])
    test_proteins = set(test_split["protein_a"]) | set(test_split["protein_b"])
    overlap = train_proteins & test_proteins
    report["original_split_audit"] = {
        "train_pairs": int(len(train_split)),
        "test_pairs": int(len(test_split)),
        "train_unique_proteins": len(train_proteins),
        "test_unique_proteins": len(test_proteins),
        "protein_overlap_count": len(overlap),
        "protein_overlap_fraction_of_test": len(overlap) / len(test_proteins) if test_proteins else 0.0,
        "note": (
            "This is the benchmark's naive random split. Protein overlap between "
            "train and test means it must only be used as the conventional "
            "baseline, never as evidence of unseen-protein generalization."
        ),
    }
    logger.info(f"Original split protein overlap: {len(overlap)}/{len(test_proteins)} test proteins also in train")

    # --- Protein-level statistics ---
    length_stats = audit.sequence_length_stats(proteins)
    degrees = audit.protein_pair_degrees(pairs, proteins)
    protein_stats = length_stats.merge(degrees, on="protein_id")
    protein_stats_path = REPO_ROOT / "results" / "protein_statistics.csv"
    protein_stats_path.parent.mkdir(parents=True, exist_ok=True)
    protein_stats.to_csv(protein_stats_path, index=False)
    logger.info(f"Wrote {protein_stats_path}")

    report["protein_statistics_summary"] = {
        "sequence_length": {
            "min": int(length_stats["length"].min()),
            "max": int(length_stats["length"].max()),
            "mean": float(length_stats["length"].mean()),
            "median": float(length_stats["length"].median()),
        },
        "amino_acid_composition": audit.amino_acid_composition(proteins),
        "proteins_with_zero_positive_degree": int((degrees["positive_degree"] == 0).sum()),
        "proteins_with_zero_negative_degree": int((degrees["negative_degree"] == 0).sum()),
    }

    # --- Pair-level statistics ---
    pairs_out = pairs.copy()
    pairs_out["protein_a_length"] = pairs_out["protein_a"].map(dict(zip(proteins["protein_id"], proteins["sequence"].str.len())))
    pairs_out["protein_b_length"] = pairs_out["protein_b"].map(dict(zip(proteins["protein_id"], proteins["sequence"].str.len())))
    pair_stats_path = REPO_ROOT / "results" / "pair_statistics.csv"
    pairs_out.to_csv(pair_stats_path, index=False)
    logger.info(f"Wrote {pair_stats_path}")

    # --- Network analysis (positive PPI graph only) ---
    g = audit.positive_ppi_graph(pairs)
    net_summary = audit.network_summary(g)
    report["network_summary"] = net_summary
    logger.info(f"Network summary: {net_summary}")

    # --- Key question: can models exploit degree instead of compatibility? ---
    pos_corr = degrees["positive_degree"].corr(degrees["negative_degree"])
    report["degree_leakage_signal"] = {
        "positive_negative_degree_correlation": float(pos_corr),
        "interpretation": (
            "High correlation suggests high-degree ('hub') proteins are heavily "
            "represented in both positive and negative pairs, which a model could "
            "exploit via protein identity/degree rather than learning genuine "
            "interaction compatibility. Should be re-examined per split in later analyses."
        ),
    }

    save_json(report, REPO_ROOT / "results" / "dataset_audit.json")
    logger.info("Wrote results/dataset_audit.json")

    write_manifest(REPO_ROOT / "results", config=config, seed=config.get("seed"), extra={"script": "01_audit_dataset.py"})

    # --- Figures ---
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Negative (0)", "Positive (1)"], [dims["n_negative"], dims["n_positive"]], color=["#888888", "#2c7fb8"])
    ax.set_ylabel("Number of pairs")
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(length_stats["length"], bins=50, color="#2c7fb8")
    ax.set_xlabel("Sequence length (aa)")
    ax.set_ylabel("Number of proteins")
    ax.set_title("Sequence length distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "sequence_length_distribution.pdf")
    plt.close(fig)

    deg_values = [d for _, d in g.degree()]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(deg_values, bins=50, color="#2c7fb8")
    ax.set_xlabel("Degree (positive-interaction network)")
    ax.set_ylabel("Number of proteins")
    ax.set_title("Network degree distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "network_degree_distribution.pdf")
    plt.close(fig)

    logger.info(f"Wrote figures to {figures_dir}")
    logger.info("Audit complete.")


if __name__ == "__main__":
    main()
