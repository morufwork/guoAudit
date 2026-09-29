"""Extract AAC/CTD/physicochemical descriptors.

Computes each descriptor once per unique protein (2,497 proteins, not
11,164 pairs) and caches to data/processed/features/. Reads from
data/processed/proteins_with_groups.tsv, not data/raw/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.features import aac, ctd, physicochemical
from src.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("extract_handcrafted_features")

EXTRACTORS = {
    "aac": (aac.compute_aac, aac.feature_names),
    "ctd": (ctd.compute_ctd, ctd.feature_names),
    "physicochemical": (physicochemical.compute_physicochemical, physicochemical.feature_names),
}


def main():
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")

    out_dir = REPO_ROOT / "data" / "processed" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for repr_name, (compute_fn, names_fn) in EXTRACTORS.items():
        cols = names_fn()
        rows = [compute_fn(seq) for seq in proteins["sequence"]]
        df = pd.DataFrame(rows, columns=cols)
        df.insert(0, "protein_id", proteins["protein_id"].values)
        assert not df[cols].isna().any().any(), f"NaN in {repr_name} features"
        assert not (df[cols].abs() == float("inf")).any().any(), f"Inf in {repr_name} features"
        out_path = out_dir / f"{repr_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote {out_path}: {df.shape[0]} proteins x {len(cols)} features")

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
