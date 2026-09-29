"""ESM-2 embedding cache integrity."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "embeddings" / "esm2"


@pytest.fixture(scope="module")
def proteins_with_groups():
    return pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")


@pytest.fixture(scope="module")
def index():
    return pd.read_csv(EMB_DIR / "index.csv")


@pytest.fixture(scope="module")
def h5_data():
    with h5py.File(EMB_DIR / "protein_embeddings.h5") as f:
        yield {
            "mean": f["mean"][:],
            "cls": f["cls"][:],
            "protein_id": [pid.decode() for pid in f["protein_id"][:]],
            "attrs": dict(f.attrs),
        }


def test_embedding_count_matches_expected_proteins(index, proteins_with_groups):
    assert len(index) == len(proteins_with_groups) == 2497
    assert set(index["protein_id"]) == set(proteins_with_groups["protein_id"])


def test_exactly_one_embedding_per_protein(index):
    assert index["protein_id"].duplicated().sum() == 0


def test_embedding_dimensions_consistent(h5_data, index):
    n, dim = h5_data["mean"].shape
    assert n == len(index)
    assert h5_data["cls"].shape == (n, dim)
    assert (index["embedding_dimension"] == dim).all()


def test_no_nan_or_inf_in_embeddings(h5_data):
    for key in ("mean", "cls"):
        arr = h5_data[key]
        assert not np.isnan(arr).any(), f"NaN in {key} embeddings"
        assert not np.isinf(arr).any(), f"Inf in {key} embeddings"


def test_sequence_hash_matches_source_data(index, proteins_with_groups):
    """Every embedding's recorded sequence_hash must match the canonicalization hash for
    that protein_id -- catches silent protein_id/sequence misalignment."""
    hash_by_id = dict(zip(proteins_with_groups["protein_id"], proteins_with_groups["sequence_hash"]))
    for _, row in index.iterrows():
        assert row["sequence_hash"] == hash_by_id[row["protein_id"]]


def test_h5_and_index_protein_ids_match_in_order(h5_data, index):
    assert h5_data["protein_id"] == list(index["protein_id"])


def test_mean_and_cls_pooling_are_distinct(h5_data):
    """Sanity check that the two pooling strategies aren't accidentally identical."""
    assert not np.allclose(h5_data["mean"][:50], h5_data["cls"][:50])


def test_truncation_count_matches_summary(index):
    import json
    summary = json.load(open(REPO_ROOT / "results" / "esm2_extraction_summary.json"))
    assert int(index["truncated"].sum()) == summary["n_truncated"]


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_load_esm2_features_matches_h5_cache(pooling, h5_data):
    from src.features.esm2 import load_esm2_features

    df = load_esm2_features(pooling, EMB_DIR)
    assert list(df["protein_id"]) == h5_data["protein_id"]
    feature_cols = [c for c in df.columns if c != "protein_id"]
    assert len(feature_cols) == h5_data[pooling].shape[1]
    np.testing.assert_array_equal(df[feature_cols].to_numpy(), h5_data[pooling])


def test_embed_sequence_is_deterministic():
    """Frozen model in eval mode: same sequence must give identical embeddings
    across calls (no dropout/randomness at inference time)."""
    from src.features.esm2 import embed_sequence, load_model
    from src.utils.io import load_config

    config = load_config(REPO_ROOT / "configs" / "esm2.yaml")
    tokenizer, model = load_model(config["model_name"])
    seq = "MKTLLILAVLLTAAVCADASGDKSGDKS"
    r1 = embed_sequence(tokenizer, model, seq, config["max_length"])
    r2 = embed_sequence(tokenizer, model, seq, config["max_length"])
    np.testing.assert_array_equal(r1["mean"], r2["mean"])
    np.testing.assert_array_equal(r1["cls"], r2["cls"])
