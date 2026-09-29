"""Amino Acid Composition (AAC): per-protein frequency of each of the 20 standard residues."""
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def compute_aac(sequence: str) -> np.ndarray:
    """20-dim vector of residue frequencies, in AMINO_ACIDS order."""
    n = len(sequence)
    return np.array([sequence.count(aa) / n for aa in AMINO_ACIDS], dtype=np.float64)


def feature_names() -> list[str]:
    return [f"aac_{aa}" for aa in AMINO_ACIDS]
