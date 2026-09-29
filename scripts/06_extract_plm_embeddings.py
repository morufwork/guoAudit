"""ESM-2 embedding extraction.

Computes each protein's embedding ONCE (2,497 proteins, not 11,164+ pairs)
and caches it. Writes:
  embeddings/esm2/protein_embeddings.h5   (datasets: mean, cls -- (N, dim))
  embeddings/esm2/index.csv               (protein_id, sequence_hash,
                                            embedding_model, embedding_dimension,
                                            pooling_method, truncated, ...)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd

from src.features.esm2 import embed_sequence, load_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("extract_plm_embeddings")


def main():
    config = load_config(REPO_ROOT / "configs" / "esm2.yaml")
    set_seed(config["seed"])

    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    logger.info(f"Loading {config['model_name']} ...")
    t0 = time.time()
    tokenizer, model = load_model(config["model_name"])
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    n = len(proteins)
    mean_embeddings = None
    cls_embeddings = None
    index_rows = []

    t_start = time.time()
    for i, row in enumerate(proteins.itertuples()):
        result = embed_sequence(tokenizer, model, row.sequence, config["max_length"])

        if mean_embeddings is None:
            dim = len(result["mean"])
            mean_embeddings = np.zeros((n, dim), dtype=np.float32)
            cls_embeddings = np.zeros((n, dim), dtype=np.float32)

        mean_embeddings[i] = result["mean"]
        cls_embeddings[i] = result["cls"]
        index_rows.append({
            "protein_id": row.protein_id,
            "sequence_hash": row.sequence_hash,
            "embedding_model": config["model_name"],
            "embedding_dimension": dim,
            "pooling_method": "mean+cls",
            "original_length": result["original_length"],
            "used_length": result["used_length"],
            "truncated": result["truncated"],
        })

        if (i + 1) % 250 == 0 or (i + 1) == n:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("inf")
            logger.info(f"{i + 1}/{n} proteins embedded ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    index_df = pd.DataFrame(index_rows)

    # --- Acceptance criteria checks ---
    assert len(index_df) == n == proteins["protein_id"].nunique(), "expected exactly one embedding per unique protein"
    assert not np.isnan(mean_embeddings).any() and not np.isinf(mean_embeddings).any(), "NaN/Inf in mean embeddings"
    assert not np.isnan(cls_embeddings).any() and not np.isinf(cls_embeddings).any(), "NaN/Inf in cls embeddings"
    assert (index_df["sequence_hash"].notna()).all(), "every embedding must have an associated sequence hash"

    n_truncated = int(index_df["truncated"].sum())
    logger.info(f"NaN/Inf check passed. {n_truncated}/{n} proteins truncated to {config['max_length']} residues.")

    out_dir = REPO_ROOT / "embeddings" / "esm2"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_dir / "protein_embeddings.h5", "w") as f:
        f.create_dataset("mean", data=mean_embeddings)
        f.create_dataset("cls", data=cls_embeddings)
        f.create_dataset("protein_id", data=np.array(index_df["protein_id"], dtype=h5py.string_dtype()))
        f.attrs["embedding_model"] = config["model_name"]
        f.attrs["embedding_dimension"] = dim
        f.attrs["max_length"] = config["max_length"]

    index_df.to_csv(out_dir / "index.csv", index=False)
    logger.info(f"Wrote {out_dir}/protein_embeddings.h5 and index.csv ({n} proteins x {dim} dims, mean+cls pooling)")

    save_json(
        {
            "n_proteins": n,
            "embedding_dimension": dim,
            "model_name": config["model_name"],
            "max_length": config["max_length"],
            "n_truncated": n_truncated,
            "truncated_protein_ids": index_df.loc[index_df["truncated"], "protein_id"].tolist(),
            "total_extraction_seconds": time.time() - t_start,
        },
        REPO_ROOT / "results" / "esm2_extraction_summary.json",
    )
    write_manifest(REPO_ROOT / "results", config=config, seed=config["seed"], extra={"script": "06_extract_plm_embeddings.py"})
    logger.info("ESM-2 extraction complete.")


if __name__ == "__main__":
    main()
