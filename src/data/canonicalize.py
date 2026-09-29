"""Canonical (undirected) pair representation.

Best practice for an undirected PPI benchmark: store every
pair with a fixed, order-independent ordering so A-B and B-A can never be
treated as different examples, then drop exact duplicate rows that result.

The 24 duplicate pairs found in the dataset audit carried no label conflicts,
so dropping the redundant copy is safe — it removes double-weighting, not
information.
"""
import pandas as pd


def canonicalize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Reorder every row so protein_a < protein_b lexicographically, then
    drop exact duplicate rows. Does not mutate the input."""
    out = pairs.copy()
    swap = out["protein_a"] > out["protein_b"]
    out.loc[swap, ["protein_a", "protein_b"]] = out.loc[swap, ["protein_b", "protein_a"]].values
    before = len(out)
    out = out.drop_duplicates(subset=["protein_a", "protein_b"], keep="first").reset_index(drop=True)
    after = len(out)
    out.attrs["n_dropped_duplicates"] = before - after
    return out
