"""CTD (Composition, Transition, Distribution) descriptors.

Classical Dubchak/PROFEAT scheme: 7 physicochemical attributes, each
partitioning the 20 standard amino acids into 3 groups. For each attribute:
  - Composition (C): 3 features — fraction of residues in each group.
  - Transition (T): 3 features — fraction of adjacent-residue transitions
    between each pair of groups.
  - Distribution (D): 15 features — for each group, the sequence position
    (as a fraction of length) at which the 1st/25%/50%/75%/100th occurrence
    of that group is reached.
Total: 7 * (3 + 3 + 15) = 147 features.

Group definitions are fixed literature lookup tables (not fit from data),
so computing this introduces no leakage risk.
"""
import numpy as np

# attribute -> (group1, group2, group3), each a string of amino acids.
# Groups partition the 20 standard amino acids exactly once each.
ATTRIBUTES = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "normalized_vdw_volume": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary_structure": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent_accessibility": ("ALFCGIVW", "RKQEND", "MSPTHY"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
}

for _name, (_g1, _g2, _g3) in ATTRIBUTES.items():
    assert len(_g1) + len(_g2) + len(_g3) == 20, _name
    assert set(_g1) | set(_g2) | set(_g3) == set("ACDEFGHIKLMNPQRSTVWY"), _name


def _group_index(groups: tuple[str, str, str]) -> dict[str, int]:
    idx = {}
    for g_id, group in enumerate(groups, start=1):
        for aa in group:
            idx[aa] = g_id
    return idx


def _composition(coded: np.ndarray) -> list[float]:
    n = len(coded)
    return [(coded == g).sum() / n for g in (1, 2, 3)]


def _transition(coded: np.ndarray) -> list[float]:
    n = len(coded)
    if n < 2:
        return [0.0, 0.0, 0.0]
    pairs = list(zip(coded[:-1], coded[1:]))
    n_transitions = n - 1
    t12 = sum(1 for a, b in pairs if {a, b} == {1, 2}) / n_transitions
    t13 = sum(1 for a, b in pairs if {a, b} == {1, 3}) / n_transitions
    t23 = sum(1 for a, b in pairs if {a, b} == {2, 3}) / n_transitions
    return [t12, t13, t23]


def _distribution(coded: np.ndarray) -> list[float]:
    n = len(coded)
    features = []
    for g in (1, 2, 3):
        positions = np.where(coded == g)[0]
        if len(positions) == 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        count = len(positions)
        checkpoints = [
            positions[0],
            positions[max(int(count * 0.25) - 1, 0)],
            positions[max(int(count * 0.50) - 1, 0)],
            positions[max(int(count * 0.75) - 1, 0)],
            positions[-1],
        ]
        features.extend([(p + 1) / n for p in checkpoints])
    return features


def compute_ctd(sequence: str) -> np.ndarray:
    seq_arr = np.array(list(sequence))
    features = []
    for attr, groups in ATTRIBUTES.items():
        idx = _group_index(groups)
        coded = np.array([idx[aa] for aa in seq_arr])
        features.extend(_composition(coded))
        features.extend(_transition(coded))
        features.extend(_distribution(coded))
    return np.array(features, dtype=np.float64)


def feature_names() -> list[str]:
    names = []
    for attr in ATTRIBUTES:
        names.extend([f"ctd_{attr}_C{g}" for g in (1, 2, 3)])
        names.extend([f"ctd_{attr}_T{g}" for g in ("12", "13", "23")])
        for g in (1, 2, 3):
            names.extend([f"ctd_{attr}_D{g}_{p}" for p in (0, 25, 50, 75, 100)])
    return names
