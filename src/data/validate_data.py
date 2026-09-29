"""Integrity checks for the Guo yeast benchmark.

Each function returns plain data (counts / DataFrames of offending rows) so
callers can both assert on it in tests and serialize it into the audit report.
"""
import pandas as pd


def canonical_pair(row) -> tuple[str, str]:
    a, b = row["protein_a"], row["protein_b"]
    return (a, b) if a < b else (b, a)


def missing_sequences(proteins: pd.DataFrame) -> pd.DataFrame:
    return proteins[proteins["sequence"].isna() | (proteins["sequence"].str.len() == 0)]


def missing_ids(proteins: pd.DataFrame) -> pd.DataFrame:
    return proteins[proteins["protein_id"].isna() | (proteins["protein_id"].str.len() == 0)]


def invalid_amino_acids(proteins: pd.DataFrame, valid_aa: str) -> pd.DataFrame:
    valid = set(valid_aa)
    mask = proteins["sequence"].apply(lambda s: not set(str(s)).issubset(valid))
    return proteins[mask]


def duplicate_protein_ids(proteins: pd.DataFrame) -> pd.DataFrame:
    return proteins[proteins["protein_id"].duplicated(keep=False)]


def identical_sequences_diff_ids(proteins: pd.DataFrame) -> pd.DataFrame:
    dup = proteins[proteins["sequence"].duplicated(keep=False)]
    return dup.sort_values("sequence")


def duplicated_pairs_exact(pairs: pd.DataFrame) -> pd.DataFrame:
    mask = pairs.duplicated(subset=["protein_a", "protein_b"], keep=False)
    return pairs[mask]


def reverse_duplicate_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows whose canonicalized (A,B) form appears more than once, including
    reverse-order duplicates (A,B) vs (B,A) as well as exact duplicates."""
    canon = pairs.apply(canonical_pair, axis=1)
    counts = canon.value_counts()
    dup_keys = set(counts[counts > 1].index)
    mask = canon.apply(lambda k: k in dup_keys)
    return pairs[mask]


def conflicting_labels(pairs: pd.DataFrame) -> pd.DataFrame:
    canon = pairs.apply(canonical_pair, axis=1)
    tmp = pairs.assign(_canon=canon)
    grouped = tmp.groupby("_canon")["label"].nunique()
    conflicting_keys = set(grouped[grouped > 1].index)
    return tmp[tmp["_canon"].isin(conflicting_keys)].drop(columns="_canon")


def self_interactions(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs[pairs["protein_a"] == pairs["protein_b"]]


def pairs_referencing_unknown_proteins(pairs: pd.DataFrame, proteins: pd.DataFrame) -> pd.DataFrame:
    known = set(proteins["protein_id"])
    mask = ~pairs["protein_a"].isin(known) | ~pairs["protein_b"].isin(known)
    return pairs[mask]
