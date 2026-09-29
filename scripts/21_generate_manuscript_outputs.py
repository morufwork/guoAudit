"""Manuscript output consolidation: tables are generated programmatically
from saved results. Builds Table 1 (dataset characteristics) purely from JSON/CSV
already written by the dataset audit (audit), the canonicalization (canonicalization), and the sequence-cluster split
(homology clustering) -- no experiments are re-run, nothing is recomputed.

Also writes an explicit, machine-readable status note for Table 9 (external
validation): cross-species external validation was outside the scope of this
study, so Table 9 does not exist by omission -- it is recorded as
deliberately not applicable, so that exclusions are logged in machine-readable form.

Writes:
  tables/table1_dataset_characteristics.csv
  tables/table9_external_validation_status.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.io import save_json
from src.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("generate_manuscript_outputs")


def _fraction_with_any_homolog(cluster_tsv: Path) -> float:
    """Fraction of proteins sharing a cluster with >=1 other protein.

    NOT `1 - n_clusters/n_proteins` -- that conflates cluster-count
    reduction with protein membership and was the source of a real error
    caught 2026-08-23 (the sequence-cluster split originally reported ~7.7% at 20% identity;
    the true value must be computed directly from the current, versioned
    cluster membership rather than frozen as a manuscript constant). Must be computed
    directly from cluster assignment, as here."""
    df = pd.read_csv(cluster_tsv, sep="\t")
    cluster_sizes = df.groupby("cluster_id").size()
    n_in_nonsingleton = cluster_sizes[cluster_sizes > 1].sum()
    return n_in_nonsingleton / len(df)


def build_table1():
    audit = json.load(open(REPO_ROOT / "results" / "dataset_audit.json"))
    canon = json.load(open(REPO_ROOT / "results" / "canonicalization_report.json"))

    dims = audit["dimensions"]
    net = audit["network_summary"]

    # Fraction of proteins with >=1 homolog at the loosest (20%) identity
    # threshold checked -- see _fraction_with_any_homolog's docstring for
    # why this must come from cluster membership, not cluster counts.
    loosest = "20"
    frac_with_homolog = _fraction_with_any_homolog(
        REPO_ROOT / "data" / "processed" / "clusters" / f"cluster_{loosest}.tsv"
    )

    rows = [
        ("Unique proteins (raw)", dims["n_proteins"]),
        ("PPI pairs (raw)", dims["n_pairs"]),
        ("Positive pairs (raw)", dims["n_positive"]),
        ("Negative pairs (raw, N0)", dims["n_negative"]),
        ("Canonical (deduplicated) pairs", canon["canonical_pair_count"]),
        ("Exact duplicate pairs removed", canon["exact_duplicate_pairs_dropped"]),
        ("Sequence-identity equivalence groups", canon["n_sequence_groups"]),
        ("Non-singleton identical-sequence groups", len(canon["non_singleton_sequence_groups"])),
        ("Positive-interaction graph: nodes", net["n_nodes"]),
        ("Positive-interaction graph: edges", net["n_edges"]),
        ("Positive-interaction graph: density", round(net["density"], 5)),
        ("Positive-interaction graph: mean degree", round(net["mean_degree"], 2)),
        ("Positive-interaction graph: max degree", net["max_degree"]),
        ("Positive-interaction graph: connected components", net["n_connected_components"]),
        ("Positive-interaction graph: largest component size", net["largest_connected_component_size"]),
        ("Clustering coefficient", round(net["average_clustering_coefficient"], 4)),
        (f"Proteins with >=1 homolog at {loosest}% identity (fraction)", round(frac_with_homolog, 4)),
    ]
    df = pd.DataFrame(rows, columns=["Characteristic", "Value"])
    out_path = REPO_ROOT / "tables" / "table1_dataset_characteristics.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}")


def build_table9_status():
    status = {
        "table": "Table 9 -- Optional external (cross-species) validation",
        "status": "not_applicable_by_design",
        "reason": (
            "Cross-species external validation was outside the scope of this study: "
            "it depends on a working predictive model, "
            "and the architecture ablation and statistical analysis established that no architecture tried clears chance "
            "under the confound-controlled (N2) regime on this benchmark. Recorded "
            "explicitly rather than silently omitted."
        ),
        "decided": "2026-08-23",
    }
    out_path = REPO_ROOT / "tables" / "table9_external_validation_status.json"
    save_json(status, out_path)
    logger.info(f"Wrote {out_path}")


def main():
    build_table1()
    build_table9_status()
    logger.info("Manuscript output consolidation complete.")


if __name__ == "__main__":
    main()
