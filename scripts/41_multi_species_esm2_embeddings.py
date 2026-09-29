"""ESM-2 embeddings for the multi-species
replication's nonlinear sequence model (direct comparability with the
yeast headline "ESM-2 + MLP" model requires the same checkpoint/pooling/
truncation choices -- Section 2.4). Same model and max_length as
scripts/06 (facebook/esm2_t12_35M_UR50D, max_length=1022); this dataset
has ~4.6x as many proteins (11,529 vs. 2,497) and a longer tail (max
18,074 vs. 4,910 aa, all truncated the same way), so this is the heaviest
compute step in this replication -- expect roughly 4-5x scripts/06's
runtime on this project's CPU-only environment.

Writes:
  embeddings/esm2_multi_species/protein_embeddings.h5  (dataset: mean -- (N, dim))
  embeddings/esm2_multi_species/index.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd

from src.features.esm2 import embed_sequence, load_model
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("multi_species_esm2_embeddings")

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MAX_LENGTH = 1022
SEED = 42


def main():
    set_seed(SEED)
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "multi_species" / "proteins_with_groups.tsv", sep="\t")
    logger.info(f"Loading {MODEL_NAME} ...")
    t0 = time.time()
    tokenizer, model = load_model(MODEL_NAME)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    n = len(proteins)
    mean_embeddings = None
    index_rows = []

    t_start = time.time()
    for i, row in enumerate(proteins.itertuples()):
        result = embed_sequence(tokenizer, model, row.sequence, MAX_LENGTH)
        if mean_embeddings is None:
            dim = len(result["mean"])
            mean_embeddings = np.zeros((n, dim), dtype=np.float32)
        mean_embeddings[i] = result["mean"]
        index_rows.append({
            "protein_id": row.protein_id, "sequence_hash": row.sequence_hash,
            "original_length": result["original_length"], "used_length": result["used_length"],
            "truncated": result["truncated"],
        })
        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("inf")
            logger.info(f"{i + 1}/{n} proteins embedded ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    index_df = pd.DataFrame(index_rows)
    assert len(index_df) == n == proteins["protein_id"].nunique()
    assert not np.isnan(mean_embeddings).any() and not np.isinf(mean_embeddings).any()

    n_truncated = int(index_df["truncated"].sum())
    logger.info(f"NaN/Inf check passed. {n_truncated}/{n} proteins truncated to {MAX_LENGTH} residues.")

    out_dir = REPO_ROOT / "embeddings" / "esm2_multi_species"
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / "protein_embeddings.h5", "w") as f:
        f.create_dataset("mean", data=mean_embeddings)
        f.create_dataset("protein_id", data=np.array(index_df["protein_id"], dtype=h5py.string_dtype()))
        f.attrs["embedding_model"] = MODEL_NAME
        f.attrs["embedding_dimension"] = dim
        f.attrs["max_length"] = MAX_LENGTH
    index_df.to_csv(out_dir / "index.csv", index=False)
    logger.info(f"Wrote {out_dir}/protein_embeddings.h5 and index.csv ({n} proteins x {dim} dims, mean pooling)")

    save_json(
        {"n_proteins": n, "embedding_dimension": dim, "model_name": MODEL_NAME, "max_length": MAX_LENGTH,
         "n_truncated": n_truncated, "total_extraction_seconds": time.time() - t_start},
        REPO_ROOT / "results" / "multi_species_pooled" / "esm2_extraction_summary.json",
    )
    write_manifest(out_dir, config={"model_name": MODEL_NAME, "max_length": MAX_LENGTH}, seed=SEED,
                    extra={"script": "41_multi_species_esm2_embeddings.py"})
    logger.info("Multi-species ESM-2 extraction complete.")


if __name__ == "__main__":
    main()
