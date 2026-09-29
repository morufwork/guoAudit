"""Selected physicochemical descriptors via Biopython's ProtParam."""
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

FEATURE_NAMES = [
    "molecular_weight",
    "aromaticity",
    "instability_index",
    "isoelectric_point",
    "gravy",
    "helix_fraction",
    "turn_fraction",
    "sheet_fraction",
]


def compute_physicochemical(sequence: str) -> np.ndarray:
    analysis = ProteinAnalysis(sequence)
    helix, turn, sheet = analysis.secondary_structure_fraction()
    return np.array(
        [
            analysis.molecular_weight(),
            analysis.aromaticity(),
            analysis.instability_index(),
            analysis.isoelectric_point(),
            analysis.gravy(),
            helix,
            turn,
            sheet,
        ],
        dtype=np.float64,
    )


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)
