"""Sequence-identity equivalence classes.

The dataset audit found 8 protein IDs (3 groups) with byte-identical
sequences under different UniProt accessions (paralogs). A protein-disjoint
split that puts two members of the same group on opposite sides
would leak: the model would see the exact test sequence during training
under a different ID. Every later split generator must keep each group
entirely on one side of any train/test boundary.

This groups by *exact* sequence identity (100%) — a stricter, cheaper
precursor to the MMseqs2 homology-cluster thresholds (50/40/30/20%),
not a replacement for them.
"""
import hashlib

import pandas as pd


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def assign_sequence_groups(proteins: pd.DataFrame) -> pd.DataFrame:
    """Return proteins with added `sequence_hash` and `sequence_group_id` columns.

    sequence_group_id is a dense integer id shared by every protein_id with
    an identical sequence; singleton (unique) sequences each get their own id.
    """
    out = proteins.copy()
    out["sequence_hash"] = out["sequence"].apply(sequence_hash)
    out["sequence_group_id"] = out.groupby("sequence_hash").ngroup()
    return out


def group_sizes(grouped_proteins: pd.DataFrame) -> pd.Series:
    return grouped_proteins.groupby("sequence_group_id")["protein_id"].count().sort_values(ascending=False)
