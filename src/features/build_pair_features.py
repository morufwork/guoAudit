"""Build pair-level feature matrices from protein-level features + a symmetric fusion."""
import numpy as np
import pandas as pd

from src.features.pair_fusion import ABLATION_FUSIONS


def build_pair_matrix(pairs: pd.DataFrame, protein_features: pd.DataFrame, fusion: str) -> tuple[np.ndarray, np.ndarray]:
    """
    pairs: DataFrame with columns [protein_a, protein_b, label]
    protein_features: DataFrame with column protein_id + feature columns
    fusion: key into ABLATION_FUSIONS (includes SYMMETRIC_FUSIONS plus
        the pair-fusion ablation's order-sensitive/diff+product ablation entries)
    Returns (X, y).
    """
    fusion_fn = ABLATION_FUSIONS[fusion]
    feature_cols = [c for c in protein_features.columns if c != "protein_id"]
    feat_by_id = protein_features.set_index("protein_id")[feature_cols]

    h_a = feat_by_id.loc[pairs["protein_a"]].to_numpy(dtype=np.float64)
    h_b = feat_by_id.loc[pairs["protein_b"]].to_numpy(dtype=np.float64)
    x = fusion_fn(h_a, h_b)
    y = pairs["label"].to_numpy()
    return x, y
