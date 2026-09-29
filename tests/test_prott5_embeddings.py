"""ProtT5 embedding cache integrity tests,
mirroring tests/test_embeddings.py's ESM-2 coverage for this second,
independent PLM. Requires scripts/19_extract_prott5_embeddings.py to have
been run first (skipped otherwise, same convention as test_embeddings.py
implicitly assumes via its module-scoped fixtures)."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "embeddings" / "prott5"

pytestmark = pytest.mark.skipif(
    not (EMB_DIR / "protein_embeddings.h5").exists(),
    reason="ProtT5 embeddings not extracted -- run scripts/19_extract_prott5_embeddings.py first",
)


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
    assert (index["embedding_dimension"] == dim).all()


def test_no_nan_or_inf_in_embeddings(h5_data):
    arr = h5_data["mean"]
    assert not np.isnan(arr).any(), "NaN in mean embeddings"
    assert not np.isinf(arr).any(), "Inf in mean embeddings"


def test_sequence_hash_matches_source_data(index, proteins_with_groups):
    hash_by_id = dict(zip(proteins_with_groups["protein_id"], proteins_with_groups["sequence_hash"]))
    for _, row in index.iterrows():
        assert row["sequence_hash"] == hash_by_id[row["protein_id"]]


def test_h5_and_index_protein_ids_match_in_order(h5_data, index):
    assert h5_data["protein_id"] == list(index["protein_id"])


def test_truncation_count_matches_summary(index):
    import json
    summary = json.load(open(REPO_ROOT / "results" / "prott5_extraction_summary.json"))
    assert int(index["truncated"].sum()) == summary["n_truncated"]


def test_load_prott5_features_matches_h5_cache(h5_data):
    from src.features.prott5 import load_prott5_features

    df = load_prott5_features(EMB_DIR)
    assert list(df["protein_id"]) == h5_data["protein_id"]
    feature_cols = [c for c in df.columns if c != "protein_id"]
    assert len(feature_cols) == h5_data["mean"].shape[1]
    np.testing.assert_array_equal(df[feature_cols].to_numpy(), h5_data["mean"])


def test_prott5_and_esm2_embeddings_are_not_accidentally_identical(h5_data):
    """Sanity check against a copy-paste-style bug: two independent PLMs
    with different architectures/dimensions must not yield the same
    embedding space."""
    esm2_dir = REPO_ROOT / "embeddings" / "esm2"
    if not (esm2_dir / "protein_embeddings.h5").exists():
        pytest.skip("ESM-2 embeddings not present")
    with h5py.File(esm2_dir / "protein_embeddings.h5") as f:
        esm2_mean = f["mean"][:5]
        esm2_ids = [pid.decode() for pid in f["protein_id"][:5]]
    prott5_by_id = dict(zip(h5_data["protein_id"], h5_data["mean"]))
    assert h5_data["mean"].shape[1] != esm2_mean.shape[1] or not np.allclose(
        [prott5_by_id[pid] for pid in esm2_ids], esm2_mean
    )


def test_embed_sequence_is_deterministic():
    """Frozen model in eval mode: same sequence must give identical embeddings
    across calls (no dropout/randomness at inference time)."""
    from src.features.prott5 import embed_sequence, load_model
    from src.utils.io import load_config

    config = load_config(REPO_ROOT / "configs" / "prott5.yaml")
    tokenizer, model = load_model(config["model_name"])
    seq = "MKTLLILAVLLTAAVCADASGDKSGDKS"
    r1 = embed_sequence(tokenizer, model, seq, config["max_length"])
    r2 = embed_sequence(tokenizer, model, seq, config["max_length"])
    np.testing.assert_array_equal(r1["mean"], r2["mean"])


def test_embed_sequence_maps_rare_residues_to_x():
    """ProtTrans convention: U/Z/O/B are mapped to X before tokenization --
    verified here by confirming a sequence containing U embeds identically
    to the same sequence with U pre-replaced by X."""
    from src.features.prott5 import embed_sequence, load_model
    from src.utils.io import load_config

    config = load_config(REPO_ROOT / "configs" / "prott5.yaml")
    tokenizer, model = load_model(config["model_name"])
    seq_with_u = "MKTLLUAVLLTAAVCADASGDKS"
    seq_with_x = "MKTLLXAVLLTAAVCADASGDKS"
    r1 = embed_sequence(tokenizer, model, seq_with_u, config["max_length"])
    r2 = embed_sequence(tokenizer, model, seq_with_x, config["max_length"])
    np.testing.assert_array_equal(r1["mean"], r2["mean"])
