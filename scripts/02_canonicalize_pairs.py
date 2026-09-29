"""Canonical pair representation + sequence-identity groups.

Consumes the raw dataset (data/raw/), never modifies it. Writes:
  data/processed/ppi_pairs_clean.tsv     — canonicalized, deduplicated pairs
  data/processed/proteins_with_groups.tsv — proteins + sequence_hash + sequence_group_id
  results/canonicalization_report.json

Downstream steps (splits, features, models) must read from data/processed/,
not data/raw/, so the dedup/grouping decisions are applied exactly once and
are auditable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.canonicalize import canonicalize_pairs
from src.data.load_data import load_pairs, load_proteins
from src.data.sequence_groups import assign_sequence_groups, group_sizes
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("canonicalize_pairs")


def main():
    config = load_config(REPO_ROOT / "configs" / "data.yaml")
    proteins = load_proteins(config, REPO_ROOT)
    pairs = load_pairs(config, REPO_ROOT)

    # --- Canonical pair ordering + dedup ---
    clean_pairs = canonicalize_pairs(pairs)
    n_dropped = clean_pairs.attrs["n_dropped_duplicates"]
    assert (clean_pairs["protein_a"] < clean_pairs["protein_b"]).all(), "canonicalization invariant violated"

    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_pairs_path = processed_dir / "ppi_pairs_clean.tsv"
    clean_pairs.to_csv(clean_pairs_path, sep="\t", header=False, index=False)
    logger.info(f"Wrote {clean_pairs_path}: {len(pairs)} raw rows -> {len(clean_pairs)} canonical rows "
                f"({n_dropped} exact duplicates dropped)")

    # --- Sequence-identity equivalence groups ---
    grouped = assign_sequence_groups(proteins)
    grouped_path = processed_dir / "proteins_with_groups.tsv"
    grouped.to_csv(grouped_path, sep="\t", index=False)
    sizes = group_sizes(grouped)
    multi_member_groups = sizes[sizes > 1]
    logger.info(f"Wrote {grouped_path}: {grouped['sequence_group_id'].nunique()} sequence groups "
                f"for {len(grouped)} proteins ({len(multi_member_groups)} non-singleton groups)")

    report = {
        "raw_pair_count": int(len(pairs)),
        "canonical_pair_count": int(len(clean_pairs)),
        "exact_duplicate_pairs_dropped": int(n_dropped),
        "canonical_pair_label_balance": {
            "positive": int((clean_pairs["label"] == 1).sum()),
            "negative": int((clean_pairs["label"] == 0).sum()),
        },
        "n_proteins": int(len(grouped)),
        "n_sequence_groups": int(grouped["sequence_group_id"].nunique()),
        "non_singleton_sequence_groups": {
            str(gid): grouped.loc[grouped["sequence_group_id"] == gid, "protein_id"].tolist()
            for gid in multi_member_groups.index
        },
        "policy": (
            "Pairs are stored with protein_a < protein_b; "
            "exact duplicate rows dropped, keep-first, no label conflicts existed. "
            "Sequence groups must be kept intact (never split across train/test) "
            "by every split generator from the split generation onward."
        ),
    }
    save_json(report, REPO_ROOT / "results" / "canonicalization_report.json")
    logger.info("Wrote results/canonicalization_report.json")

    write_manifest(REPO_ROOT / "results", config=config, seed=config.get("seed"), extra={"script": "02_canonicalize_pairs.py"})
    logger.info("Canonicalization complete.")


if __name__ == "__main__":
    main()
