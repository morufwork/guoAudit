"""AAC/CTD handcrafted features for the
multi-species replication (Logistic Regression input; fast, CPU-only,
no PLM extraction needed -- see scripts/41 for the ESM-2 embeddings used
by this replication's nonlinear sequence model).

Writes: data/processed/multi_species/features/{aac,ctd}.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.features import aac, ctd
from src.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("multi_species_handcrafted_features")

EXTRACTORS = {"aac": (aac.compute_aac, aac.feature_names), "ctd": (ctd.compute_ctd, ctd.feature_names)}


def main():
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "multi_species" / "proteins_with_groups.tsv", sep="\t")
    out_dir = REPO_ROOT / "data" / "processed" / "multi_species" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for repr_name, (compute_fn, names_fn) in EXTRACTORS.items():
        cols = names_fn()
        rows = [compute_fn(seq) for seq in proteins["sequence"]]
        df = pd.DataFrame(rows, columns=cols)
        df.insert(0, "protein_id", proteins["protein_id"].values)
        assert not df[cols].isna().any().any(), f"NaN in {repr_name} features"
        out_path = out_dir / f"{repr_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote {out_path}: {df.shape[0]} proteins x {len(cols)} features")

    logger.info("Multi-species handcrafted feature extraction complete.")


if __name__ == "__main__":
    main()
