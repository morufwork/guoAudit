"""ProtT5 frozen feature extraction (Tier-2 independent-PLM cross-check).

Sequence -> ProtT5 encoder (frozen, no fine-tuning) -> residue embeddings ->
mean pooling -> one protein embedding. Computed once per unique protein,
mirroring src/features/esm2.py's extraction procedure exactly.

Model choice: `Rostlab/prot_t5_base_mt_uniref50` (T5-base scale, ~85M
encoder parameters), not the commonly-cited ProtT5-XL-UniRef50 (~3B
parameters). Timed at load time on this project's CPU-only, 4-core,
7.2GB-RAM environment (the same constraint documented in configs/esm2.yaml):
the XL checkpoint's weights alone would consume more memory than is
available on this machine, and per-sequence forward-pass cost would be
orders of magnitude slower. The base checkpoint (~0.5s/sequence at 500aa,
~1.5s at the 1022aa cap) is directly comparable in tractability to the ESM-2 extraction's
ESM-2 t12_35M choice, made for the identical documented reason.

ProtT5 (a T5 encoder) has no CLS-equivalent special token the way ESM/BERT
do -- only a trailing EOS is appended -- so only mean pooling is
implemented here, consistent with the representation comparison's own finding that mean pooling
was already the recommended default representation.
"""
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer


def load_prott5_features(embeddings_dir: str | Path = "embeddings/prott5") -> pd.DataFrame:
    """Loads the ProtT5 embedding cache as a protein_id + feature-columns
    DataFrame, in the same shape src/features/build_pair_features.py expects."""
    embeddings_dir = Path(embeddings_dir)
    with h5py.File(embeddings_dir / "protein_embeddings.h5") as f:
        vectors = f["mean"][:]
        protein_ids = [pid.decode() for pid in f["protein_id"][:]]
    cols = [f"prott5_mean_{i}" for i in range(vectors.shape[1])]
    df = pd.DataFrame(vectors, columns=cols)
    df.insert(0, "protein_id", protein_ids)
    return df


def load_model(model_name: str):
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _prepare_sequence(sequence: str) -> str:
    """ProtTrans convention: rare/ambiguous residues (U, Z, O, B) are mapped
    to X, and residues are space-separated (the tokenizer's vocabulary is
    per-residue, not per-token-run)."""
    return " ".join(list(re.sub(r"[UZOB]", "X", sequence)))


def embed_sequence(tokenizer, model, sequence: str, max_length: int) -> dict:
    """Returns the mean-pooled embedding for one sequence, plus truncation
    bookkeeping. T5 tokenizes as [residue_1..residue_L, EOS] -- no leading
    special token -- so mean pooling excludes only the trailing EOS."""
    original_length = len(sequence)
    truncated = original_length > max_length
    tokens = tokenizer(
        _prepare_sequence(sequence), return_tensors="pt", truncation=True, max_length=max_length + 1  # +1 for EOS
    )
    used_length = tokens["input_ids"].shape[1] - 1

    with torch.no_grad():
        output = model(**tokens)
    hidden = output.last_hidden_state[0]  # (seq_len_with_eos, hidden_dim)

    mean_pooled = hidden[:-1].mean(dim=0).numpy().astype(np.float32)

    return {
        "mean": mean_pooled,
        "original_length": original_length,
        "used_length": used_length,
        "truncated": truncated,
    }
