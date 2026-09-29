"""ESM-2 frozen feature extraction.

Sequence -> ESM-2 (frozen, no fine-tuning) -> residue embeddings -> pooling
-> one protein embedding. Computed once per unique protein (2,497), never
per PPI pair.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def load_esm2_features(pooling: str, embeddings_dir: str | Path = "embeddings/esm2") -> pd.DataFrame:
    """Loads the ESM-2 embedding cache as a protein_id + feature-columns
    DataFrame, in the same shape src/features/build_pair_features.py expects."""
    embeddings_dir = Path(embeddings_dir)
    with h5py.File(embeddings_dir / "protein_embeddings.h5") as f:
        vectors = f[pooling][:]
        protein_ids = [pid.decode() for pid in f["protein_id"][:]]
    cols = [f"esm2_{pooling}_{i}" for i in range(vectors.shape[1])]
    df = pd.DataFrame(vectors, columns=cols)
    df.insert(0, "protein_id", protein_ids)
    return df


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_sequence_variant(tokenizer, model, sequence: str, strategy: str, max_length: int = 1022) -> np.ndarray:
    """Alternate truncation/pooling strategies for
    the 268/2,497 proteins exceeding `max_length`, isolating whether Section
    2.4's C-terminal-truncation choice (keep the first `max_length` residues)
    -- as opposed to truncation itself, or ESM-2 more generally -- drives any
    of this study's ESM-2 results. Mean-pooling only (this study's primary
    representation everywhere it is used as "ESM-2"). Strategies:
      "n_terminal"   keep the first `max_length` residues (this study's
                     existing default, `embed_sequence` below).
      "c_terminal"   keep the LAST `max_length` residues instead -- tests
                     whether the removed C-terminal region specifically
                     (interaction domains, transmembrane/localization
                     signals, disordered regions can concentrate there)
                     carried information the default strategy discards.
      "full"         no truncation at all -- feasible on this project's
                     CPU-only environment (Section 2.4): the single longest
                     sequence in this dataset (Q12019, 4,910 aa) embeds in
                     under 6 seconds and under 1GB peak RSS.
      "segment_mean" splits the full sequence into non-overlapping
                     `max_length`-residue chunks, mean-pools residues within
                     each chunk (each chunk sees only its own local context
                     under self-attention, unlike "full"), then combines
                     chunk-level vectors via a length-weighted mean -- a
                     sliding-window/segment-pooling operationalization that
                     uses every residue without exceeding the model's
                     per-forward-pass length.
    """
    if strategy == "n_terminal":
        return embed_sequence(tokenizer, model, sequence, max_length)["mean"]

    if strategy == "c_terminal":
        cropped = sequence[-max_length:] if len(sequence) > max_length else sequence
        return embed_sequence(tokenizer, model, cropped, max_length)["mean"]

    if strategy == "full":
        tokens = tokenizer(sequence, return_tensors="pt", truncation=False)
        with torch.no_grad():
            output = model(**tokens)
        hidden = output.last_hidden_state[0]
        return hidden[1:-1].mean(dim=0).numpy().astype(np.float32)

    if strategy == "segment_mean":
        chunks = [sequence[i : i + max_length] for i in range(0, len(sequence), max_length)]
        chunk_vecs, chunk_lens = [], []
        for chunk in chunks:
            tokens = tokenizer(chunk, return_tensors="pt", truncation=False)
            with torch.no_grad():
                output = model(**tokens)
            hidden = output.last_hidden_state[0]
            chunk_vecs.append(hidden[1:-1].mean(dim=0).numpy().astype(np.float32))
            chunk_lens.append(len(chunk))
        weights = np.array(chunk_lens, dtype=np.float64) / sum(chunk_lens)
        return np.average(np.stack(chunk_vecs), axis=0, weights=weights).astype(np.float32)

    raise ValueError(f"unknown truncation strategy: {strategy}")


def embed_sequence(tokenizer, model, sequence: str, max_length: int) -> dict:
    """Returns mean- and CLS-pooled embeddings for one sequence, plus
    truncation bookkeeping. ESM tokenizes as [CLS, residue_1..residue_L, EOS];
    mean pooling excludes both special tokens."""
    original_length = len(sequence)
    truncated = original_length > max_length
    tokens = tokenizer(
        sequence, return_tensors="pt", truncation=True, max_length=max_length + 2  # +2 for CLS/EOS
    )
    used_length = tokens["input_ids"].shape[1] - 2

    with torch.no_grad():
        output = model(**tokens)
    hidden = output.last_hidden_state[0]  # (seq_len_with_specials, hidden_dim)

    mean_pooled = hidden[1:-1].mean(dim=0).numpy().astype(np.float32)
    cls_pooled = hidden[0].numpy().astype(np.float32)

    return {
        "mean": mean_pooled,
        "cls": cls_pooled,
        "original_length": original_length,
        "used_length": used_length,
        "truncated": truncated,
    }
