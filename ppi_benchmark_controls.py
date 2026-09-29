#!/usr/bin/env python3
"""
ppi_benchmark_controls.py
=========================

Single-file, self-contained implementation of the evaluation framework described in
"Complementary controls for protein reuse and negative sampling reveal
benchmark-specific signal in protein-protein interaction prediction".

It does not import anything from this repository's ``src/`` package, so it can be
copied and run on its own. Given a protein table and a labelled pair table it:

  1. audits and canonicalizes the dataset (duplicates, reversed duplicates,
     self-pairs, label conflicts, identical-sequence groups, positive-graph stats);
  2. builds protein-novelty splits R0 (random pairs) and the matched R1/R2/R3
     curve (seen-seen / one-unseen / both-unseen), plus an optional
     sequence-cluster-disjoint R3-C split, and gates every split with leakage checks;
  3. constructs alternative putative-negative sets (N1 random, N2 degree-matched,
     N3 length-matched, N5 degree+length-matched, N2-exact, N2-nearest) and
     reports endpoint-degree balance (SMD, KS, Wasserstein);
  4. trains sequence models (AAC+CTD, optionally a pretrained embedding) along the
     protein-novelty curve;
  5. runs topology-only (degree) and length-only controls;
  6. re-evaluates sequence models with each negative definition used consistently
     for training and testing (R0 and R3);
  7. quantifies uncertainty with connected-component bootstrap and
     delete-one-component jackknife intervals;
  8. writes tables, sample-level predictions, figures and a Markdown report.

Subcommands
-----------
  demo      generate a synthetic dataset with a planted degree bias and a planted
            sequence signal, then run the whole framework on it (about 1-2 min)
  run       run the framework on your own data
  validate  audit an input dataset only (no model training)

Examples
--------
  python ppi_benchmark_controls.py demo --out outputs/demo
  python ppi_benchmark_controls.py run \\
      --proteins data/raw/yeast/dictionary/protein.dictionary.tsv \\
      --pairs data/raw/yeast/actions/protein.actions.tsv --out outputs/guo_yeast
  python ppi_benchmark_controls.py run --proteins P.tsv --pairs X.tsv \\
      --embeddings embeddings/esm2/protein_embeddings.h5 --out outputs/guo_esm2

See README.md ("Input data format") for the file formats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

# Cap BLAS/OpenMP threads even if numpy was imported before this module (e.g. under
# pytest); oversubscription makes the small MLP fits dramatically slower.
threadpool_limits(limits=int(os.environ["OMP_NUM_THREADS"]))

__version__ = "1.0.0"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PLM_MAX_RESIDUES = 1022

NEG_LABELS = {
    "N0": "N0 original",
    "N1": "N1 random",
    "N2": "N2 degree-matched",
    "N3": "N3 length-matched",
    "N5": "N5 degree+length",
    "N2-exact": "N2 exact degree",
    "N2-nearest": "N2 nearest degree",
}
NEG_TICKS = {"N0": "N0\noriginal", "N1": "N1\nrandom", "N2": "N2\ndegree", "N3": "N3\nlength",
             "N5": "N5\ndeg+len", "N2-exact": "N2\nexact", "N2-nearest": "N2\nnearest"}
REGIME_LABELS = {"R0": "R0 random pairs", "R1": "R1 seen-seen", "R2": "R2 one-unseen",
                 "R3": "R3 both-unseen", "R3-C": "R3-C cluster-disjoint"}

# Categorical palette (fixed order, validated for adjacent-pair CVD separation).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# 1. Input parsing
# =============================================================================

def _looks_like_header(first_row: list[str], expected: set[str]) -> bool:
    return any(str(x).strip().lower() in expected for x in first_row)


def read_proteins(path: str | Path) -> pd.DataFrame:
    """Protein table -> DataFrame[protein_id, sequence], in file order.

    Accepts FASTA (.fa/.fasta/.faa) or a two-column TSV/CSV with or without a
    ``protein_id<TAB>sequence`` header."""
    path = Path(path)
    if path.suffix.lower() in {".fa", ".fasta", ".faa"}:
        ids, seqs, cur = [], [], []
        for line in path.read_text().splitlines():
            if line.startswith(">"):
                if ids:
                    seqs.append("".join(cur))
                ids.append(line[1:].split()[0])
                cur = []
            elif line.strip():
                cur.append(line.strip())
        if ids:
            seqs.append("".join(cur))
        df = pd.DataFrame({"protein_id": ids, "sequence": seqs})
    else:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        raw = pd.read_csv(path, sep=sep, header=None, dtype=str, keep_default_na=False)
        if _looks_like_header(raw.iloc[0].tolist(), {"protein_id", "sequence", "id", "seq"}):
            raw = raw.iloc[1:].reset_index(drop=True)
        df = raw.iloc[:, :2].copy()
        df.columns = ["protein_id", "sequence"]
    df["protein_id"] = df["protein_id"].astype(str).str.strip()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    return df.reset_index(drop=True)


def read_pairs(path: str | Path) -> pd.DataFrame:
    """Pair table -> DataFrame[protein_a, protein_b, label].

    TSV/CSV with or without a ``protein_a protein_b label`` header. The label
    column is optional: if absent, every row is treated as a positive (label 1)
    and the framework constructs all negatives itself."""
    path = Path(path)
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    raw = pd.read_csv(path, sep=sep, header=None, dtype=str, keep_default_na=False)
    if _looks_like_header(raw.iloc[0].tolist(), {"protein_a", "protein_b", "label", "id_a", "id_b"}):
        raw = raw.iloc[1:].reset_index(drop=True)
    df = raw.iloc[:, :2].copy()
    df.columns = ["protein_a", "protein_b"]
    df["label"] = raw.iloc[:, 2].astype(int).to_numpy() if raw.shape[1] >= 3 else 1
    df["protein_a"] = df["protein_a"].astype(str).str.strip()
    df["protein_b"] = df["protein_b"].astype(str).str.strip()
    return df.reset_index(drop=True)


def read_clusters(path: str | Path) -> dict[str, str]:
    """Optional ``protein_id<TAB>cluster_id`` table (e.g. MMseqs2 easy-cluster
    ``*_cluster.tsv`` output, which lists representative<TAB>member)."""
    raw = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if _looks_like_header(raw.iloc[0].tolist(), {"protein_id", "cluster_id", "representative_id"}):
        header = [h.strip().lower() for h in raw.iloc[0]]
        raw = raw.iloc[1:]
        raw.columns = header
        return dict(zip(raw["protein_id"], raw["cluster_id"]))
    # headerless MMseqs2 layout: representative, member
    return dict(zip(raw.iloc[:, 1], raw.iloc[:, 0]))


def read_embeddings(path: str | Path, dataset: str = "mean") -> pd.DataFrame:
    """Optional per-protein embedding -> DataFrame[protein_id, f0..fN].

    Formats: CSV/TSV (first column protein_id, remaining numeric), NPZ
    (arrays ``ids`` and ``X``), or HDF5 in this repository's layout
    (datasets ``protein_id`` and ``mean``/``cls``)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py
        with h5py.File(path, "r") as f:
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["protein_id"][:]]
            x = f[dataset][:]
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        ids, x = [str(i) for i in data["ids"]], data["X"]
    else:
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        ids, x = df.iloc[:, 0].astype(str).tolist(), df.iloc[:, 1:].to_numpy(dtype=np.float64)
    out = pd.DataFrame(np.asarray(x, dtype=np.float64), columns=[f"emb_{i}" for i in range(np.asarray(x).shape[1])])
    out.insert(0, "protein_id", ids)
    return out


def compute_esm2_embeddings(proteins: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Frozen ESM-2 mean-pooled embeddings (first 1,022 residues retained)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    vecs = []
    for i, seq in enumerate(proteins["sequence"]):
        tokens = tok(seq[:PLM_MAX_RESIDUES], return_tensors="pt")
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state[0]
        vecs.append(hidden[1:-1].mean(dim=0).numpy())
        if (i + 1) % 250 == 0:
            log(f"  ESM-2: embedded {i + 1}/{len(proteins)} proteins")
    out = pd.DataFrame(np.vstack(vecs), columns=[f"emb_{i}" for i in range(len(vecs[0]))])
    out.insert(0, "protein_id", proteins["protein_id"].to_numpy())
    return out


# =============================================================================
# 2. Dataset audit and canonicalization
# =============================================================================

def audit_and_canonicalize(proteins: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (proteins_with_groups, canonical_pairs, audit_report).

    Canonicalization orders each undirected pair lexically (protein_a <
    protein_b) and removes duplicate rows. Fatal problems raise ValueError."""
    report: dict = {"input": {"n_proteins": int(len(proteins)), "n_pairs": int(len(pairs)),
                              "n_positive": int((pairs["label"] == 1).sum()),
                              "n_negative": int((pairs["label"] == 0).sum())}}
    issues = {}
    dup_ids = proteins["protein_id"][proteins["protein_id"].duplicated()].unique().tolist()
    if dup_ids:
        raise ValueError(f"Duplicate protein identifiers in protein table: {dup_ids[:10]}")
    empty = proteins.loc[proteins["sequence"].str.len() == 0, "protein_id"].tolist()
    if empty:
        raise ValueError(f"Proteins with missing sequences: {empty[:10]}")
    bad_labels = sorted(set(pairs["label"].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"Labels must be 0 or 1; found {bad_labels}")
    known = set(proteins["protein_id"])
    missing = sorted((set(pairs["protein_a"]) | set(pairs["protein_b"])) - known)
    if missing:
        raise ValueError(f"{len(missing)} pair identifiers have no sequence, e.g. {missing[:10]}")

    nonstandard = proteins["sequence"].apply(lambda s: sum(c not in AMINO_ACIDS for c in s))
    issues["proteins_with_nonstandard_residues"] = int((nonstandard > 0).sum())

    self_pairs = pairs["protein_a"] == pairs["protein_b"]
    issues["self_pairs_removed"] = int(self_pairs.sum())
    work = pairs.loc[~self_pairs].copy()

    swap = work["protein_a"] > work["protein_b"]
    issues["rows_reordered_to_canonical"] = int(swap.sum())
    work.loc[swap, ["protein_a", "protein_b"]] = work.loc[swap, ["protein_b", "protein_a"]].values
    exact_dup = pairs.loc[~self_pairs].duplicated(["protein_a", "protein_b"]).sum()
    all_dup = work.duplicated(["protein_a", "protein_b"]).sum()
    issues["exact_duplicate_rows"] = int(exact_dup)
    issues["reversed_duplicate_rows"] = int(all_dup - exact_dup)

    n_labels = work.groupby(["protein_a", "protein_b"])["label"].nunique()
    conflicts = n_labels[n_labels > 1].index
    issues["conflicting_label_pairs_removed"] = int(len(conflicts))
    if len(conflicts):
        conflict_set = set(conflicts)
        work = work[[(a, b) not in conflict_set for a, b in zip(work["protein_a"], work["protein_b"])]]

    before = len(work)
    work = work.drop_duplicates(subset=["protein_a", "protein_b"], keep="first").reset_index(drop=True)
    issues["duplicate_pairs_removed"] = int(before - len(work))
    work["label"] = work["label"].astype(int)
    report["issues"] = issues

    prot = proteins.copy()
    prot["sequence_hash"] = prot["sequence"].apply(lambda s: hashlib.sha256(s.encode()).hexdigest())
    prot["sequence_group_id"] = prot.groupby("sequence_hash").ngroup()
    group_sizes = prot.groupby("sequence_group_id")["protein_id"].count()
    lengths = prot["sequence"].str.len()

    pos = work[work["label"] == 1]
    degree = pd.concat([pos["protein_a"], pos["protein_b"]]).value_counts().reindex(prot["protein_id"], fill_value=0)
    report["canonical"] = {
        "n_pairs": int(len(work)), "n_positive": int((work["label"] == 1).sum()),
        "n_negative": int((work["label"] == 0).sum()),
        "n_proteins_in_pairs": int(len(set(work["protein_a"]) | set(work["protein_b"]))),
        "identical_sequence_groups_size_gt1": int((group_sizes > 1).sum()),
        "proteins_in_identical_sequence_groups": int(group_sizes[group_sizes > 1].sum()),
        "sequence_length_median": float(lengths.median()),
        f"sequences_longer_than_{PLM_MAX_RESIDUES}": int((lengths > PLM_MAX_RESIDUES).sum()),
        "proteins_with_zero_positive_degree": int((degree == 0).sum()),
    }
    report["positive_graph"] = positive_graph_stats(pos)
    return prot, work, report


def connected_components(edges: pd.DataFrame) -> list[set[str]]:
    """Connected components of an undirected edge list (union-find)."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(edges["protein_a"], edges["protein_b"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps: dict[str, set[str]] = {}
    for node in list(parent):
        comps.setdefault(find(node), set()).add(node)
    return sorted(comps.values(), key=len, reverse=True)


def positive_graph_stats(pos: pd.DataFrame) -> dict:
    if pos.empty:
        return {}
    deg = pd.concat([pos["protein_a"], pos["protein_b"]]).value_counts()
    n, m = len(deg), len(pos)
    comps = connected_components(pos)
    return {"nodes": int(n), "edges": int(m), "density": round(2 * m / (n * (n - 1)), 5),
            "mean_degree": round(float(deg.mean()), 3), "max_degree": int(deg.max()),
            "connected_components": len(comps), "largest_component": len(comps[0])}


# =============================================================================
# 3. Protein-novelty and sequence-cluster splits with leakage gates
# =============================================================================

def partition_protein_pool(proteins: pd.DataFrame, test_fraction: float, seed: int,
                           group_col: str = "sequence_group_id", sort_groups: bool = False):
    """Split proteins into (train_pool, test_pool) at the group level so that no
    identical-sequence group (or sequence cluster) straddles the boundary."""
    group_ids = proteins[group_col].unique()
    if sort_groups:
        group_ids = np.asarray(sorted(group_ids))
    shuffled = np.random.default_rng(seed).permutation(group_ids)
    n_test = int(round(len(shuffled) * test_fraction))
    test_groups, train_groups = set(shuffled[:n_test]), set(shuffled[n_test:])
    train_pool = set(proteins.loc[proteins[group_col].isin(train_groups), "protein_id"])
    test_pool = set(proteins.loc[proteins[group_col].isin(test_groups), "protein_id"])
    return train_pool, test_pool


def _pool_membership(pairs: pd.DataFrame, train_pool: set, test_pool: set) -> pd.Series:
    a_tr, b_tr = pairs["protein_a"].isin(train_pool), pairs["protein_b"].isin(train_pool)
    a_te, b_te = pairs["protein_a"].isin(test_pool), pairs["protein_b"].isin(test_pool)
    cat = pd.Series("unassigned", index=pairs.index)
    cat[a_tr & b_tr] = "both_train"
    cat[a_te & b_te] = "both_test"
    cat[(a_tr & b_te) | (a_te & b_tr)] = "mixed"
    return cat


def _degree_safe_holdout(both_train: pd.DataFrame, fraction: float, seed: int):
    """Hold out R1 pairs without removing the last training occurrence of any protein."""
    degree = pd.concat([both_train["protein_a"], both_train["protein_b"]]).value_counts().to_dict()
    target = int(round(len(both_train) * fraction))
    held = []
    for idx in np.random.default_rng(seed).permutation(both_train.index):
        if len(held) >= target:
            break
        a, b = both_train.at[idx, "protein_a"], both_train.at[idx, "protein_b"]
        if degree[a] > 1 and degree[b] > 1:
            degree[a] -= 1
            degree[b] -= 1
            held.append(idx)
    held = set(held)
    mask = both_train.index.isin(held)
    return both_train.loc[~mask].reset_index(drop=True), both_train.loc[mask].reset_index(drop=True)


def make_splits(pairs: pd.DataFrame, proteins: pd.DataFrame, seed: int, test_fraction: float = 0.2,
                r1_fraction: float = 0.2, clusters: dict | None = None) -> dict:
    """R0 random pairs; R1/R2/R3 from one protein-pool partition sharing one
    training set; optional R3-C from a sequence-cluster partition."""
    splits = {}
    strat = pairs["label"] if pairs["label"].nunique() > 1 else None
    tr, te = train_test_split(pairs, test_size=test_fraction, stratify=strat, random_state=seed)
    splits["R0"] = (tr.reset_index(drop=True), te.reset_index(drop=True))

    train_pool, test_pool = partition_protein_pool(proteins, test_fraction, seed)
    cat = _pool_membership(pairs, train_pool, test_pool)
    both_train = pairs[cat == "both_train"]
    r2 = pairs[cat == "mixed"].reset_index(drop=True)
    r3 = pairs[cat == "both_test"].reset_index(drop=True)
    shared_train, r1 = _degree_safe_holdout(both_train, r1_fraction, seed)
    seen = set(shared_train["protein_a"]) | set(shared_train["protein_b"])
    train_side = r2["protein_a"].where(r2["protein_a"].isin(train_pool), r2["protein_b"])
    r2 = r2[train_side.isin(seen)].reset_index(drop=True)
    splits.update({"R1": (shared_train, r1), "R2": (shared_train, r2), "R3": (shared_train, r3)})

    if clusters:
        prot_c = proteins[["protein_id"]].copy()
        prot_c["cluster_id"] = prot_c["protein_id"].map(clusters).fillna(prot_c["protein_id"])
        tp, sp = partition_protein_pool(prot_c, test_fraction, seed, group_col="cluster_id", sort_groups=True)
        cat_c = _pool_membership(pairs, tp, sp)
        splits["R3-C"] = (pairs[cat_c == "both_train"].reset_index(drop=True),
                          pairs[cat_c == "both_test"].reset_index(drop=True))
    return splits


def _canon(df: pd.DataFrame) -> set:
    return {(a, b) if a < b else (b, a) for a, b in zip(df["protein_a"], df["protein_b"])}


def _proteins_of(df: pd.DataFrame) -> set:
    return set(df["protein_a"]) | set(df["protein_b"])


def leakage_checks(name: str, train: pd.DataFrame, test: pd.DataFrame, id_to_group: dict,
                   id_to_cluster: dict | None = None) -> dict:
    """Pair, reversed-pair, protein, sequence-group (and cluster) overlap gates.
    Raises RuntimeError if a required check fails."""
    need_disjoint = name in {"R3", "R3-C"}
    tr_p, te_p = _proteins_of(train), _proteins_of(test)
    res = {
        "exact_pair_overlap": len(set(zip(train.protein_a, train.protein_b)) & set(zip(test.protein_a, test.protein_b))),
        "reversed_pair_overlap": len(_canon(train) & _canon(test)),
        "protein_overlap": len(tr_p & te_p),
        "sequence_group_overlap": len({id_to_group[p] for p in tr_p} & {id_to_group[p] for p in te_p}),
        "labels_valid": bool(set(train.label) | set(test.label) <= {0, 1}),
    }
    if name == "R3-C" and id_to_cluster is not None:
        res["cluster_overlap"] = len({id_to_cluster.get(p, p) for p in tr_p} & {id_to_cluster.get(p, p) for p in te_p})
    failed = [k for k in ("exact_pair_overlap", "reversed_pair_overlap") if res[k]]
    if need_disjoint:
        failed += [k for k in ("protein_overlap", "sequence_group_overlap", "cluster_overlap") if res.get(k)]
    if name == "R1" and not te_p <= tr_p:
        failed.append("R1_test_protein_not_seen")
    if name == "R2":
        n_seen = test["protein_a"].isin(tr_p).astype(int) + test["protein_b"].isin(tr_p).astype(int)
        if not (n_seen == 1).all():
            failed.append("R2_not_exactly_one_seen")
    if not res["labels_valid"]:
        failed.append("labels_valid")
    res["passed"] = not failed
    if failed:
        raise RuntimeError(f"Split {name} failed leakage checks: {failed}")
    return res


# =============================================================================
# 4. Putative-negative construction and balance diagnostics
# =============================================================================

def compute_protein_stats(proteins: pd.DataFrame, positive_pairs: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """protein_id -> length, positive-network degree, and their decile strata."""
    degree = pd.concat([positive_pairs["protein_a"], positive_pairs["protein_b"]]).value_counts()
    st = proteins[["protein_id", "sequence"]].copy()
    st["length"] = st["sequence"].str.len()
    st["positive_degree"] = st["protein_id"].map(degree).fillna(0).astype(int)
    st["length_decile"] = pd.qcut(st["length"], n_deciles, labels=False, duplicates="drop")
    st["degree_decile"] = pd.qcut(st["positive_degree"].rank(method="first"), n_deciles, labels=False, duplicates="drop")
    return st.drop(columns="sequence").set_index("protein_id")


def known_pair_set(pairs: pd.DataFrame) -> set:
    return set(zip(pairs["protein_a"], pairs["protein_b"])) | set(zip(pairs["protein_b"], pairs["protein_a"]))


def sample_random_negatives(protein_ids: list[str], known: set, n: int, seed: int) -> pd.DataFrame:
    """N1: uniformly random unobserved pairs (no self-pairs, no known positives)."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(protein_ids))
    max_pairs = len(ids) * (len(ids) - 1) // 2 - len(known) // 2
    n = min(n, max_pairs)
    chosen: list[tuple[str, str]] = []
    seen: set = set()
    while len(chosen) < n:
        for a, b in rng.choice(ids, size=(max(2 * n, 100), 2)):
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in known or key in seen:
                continue
            seen.add(key)
            chosen.append(key)
            if len(chosen) >= n:
                break
    return pd.DataFrame(chosen, columns=["protein_a", "protein_b"]).assign(label=0)


def _substitute(positive_pairs, protein_stats, known, seed, candidate_fn):
    """Shared one-sided substitution loop: for each positive (u, v) keep one
    endpoint, replace the other with a candidate from ``candidate_fn``. A
    positive with no valid candidate yields no negative (never relaxed)."""
    rng = np.random.default_rng(seed)
    negatives, seen, n_skipped = [], set(), 0
    for _, row in positive_pairs.sample(frac=1.0, random_state=seed).iterrows():
        u, v = row["protein_a"], row["protein_b"]
        fixed, to_replace = (v, u) if rng.random() < 0.5 else (u, v)
        decoy = None
        for cand in candidate_fn(to_replace, rng):
            if cand == fixed or cand == to_replace:
                continue
            key = (cand, fixed) if cand < fixed else (fixed, cand)
            if key in known or key in seen:
                continue
            decoy = cand
            break
        if decoy is None:
            n_skipped += 1
            continue
        pair = (decoy, fixed) if decoy < fixed else (fixed, decoy)
        seen.add(pair)
        negatives.append(pair)
    df = pd.DataFrame(negatives, columns=["protein_a", "protein_b"]).assign(label=0)
    df.attrs["n_skipped"] = n_skipped
    return df


def sample_matched_negatives(positive_pairs, protein_stats, known, match_cols, seed, candidate_pool=None):
    """N2 (degree decile), N3 (length decile), N5 (both): one-sided matched substitution."""
    pool = protein_stats.loc[protein_stats.index.isin(candidate_pool)] if candidate_pool is not None else protein_stats
    buckets = pool.groupby(match_cols).apply(lambda g: list(g.index), include_groups=False).to_dict()

    def candidates(p, rng):
        key = tuple(protein_stats.loc[p, match_cols]) if len(match_cols) > 1 else protein_stats.loc[p, match_cols[0]]
        c = list(buckets.get(key, []))
        rng.shuffle(c)
        return c
    return _substitute(positive_pairs, protein_stats, known, seed, candidates)


def sample_exact_degree_negatives(positive_pairs, protein_stats, known, seed, candidate_pool=None):
    """N2-exact: replacement must have exactly the same positive degree."""
    pool = protein_stats.loc[protein_stats.index.isin(candidate_pool)] if candidate_pool is not None else protein_stats
    buckets = pool.groupby("positive_degree").apply(lambda g: list(g.index), include_groups=False).to_dict()

    def candidates(p, rng):
        c = list(buckets.get(protein_stats.loc[p, "positive_degree"], []))
        rng.shuffle(c)
        return c
    return _substitute(positive_pairs, protein_stats, known, seed, candidates)


def sample_nearest_degree_negatives(positive_pairs, protein_stats, known, seed, candidate_pool=None):
    """N2-nearest: replacement with the smallest absolute degree difference
    (random order within each distance tier)."""
    pool = protein_stats.loc[protein_stats.index.isin(candidate_pool)] if candidate_pool is not None else protein_stats
    groups = pool.groupby("positive_degree").apply(lambda g: g.index.to_numpy(), include_groups=False)
    degrees, arrays = groups.index.to_numpy(), groups.to_numpy()

    def candidates(p, rng):
        dist = np.abs(degrees - protein_stats.loc[p, "positive_degree"])
        for gi in np.argsort(dist, kind="stable"):
            grp = arrays[gi]
            for j in rng.permutation(len(grp)):
                yield grp[j]
    return _substitute(positive_pairs, protein_stats, known, seed, candidates)


NEGATIVE_BUILDERS = {
    "N2": lambda pos, st, kn, seed, pool=None: sample_matched_negatives(pos, st, kn, ["degree_decile"], seed, pool),
    "N3": lambda pos, st, kn, seed, pool=None: sample_matched_negatives(pos, st, kn, ["length_decile"], seed, pool),
    "N5": lambda pos, st, kn, seed, pool=None: sample_matched_negatives(pos, st, kn, ["degree_decile", "length_decile"], seed, pool),
    "N2-exact": lambda pos, st, kn, seed, pool=None: sample_exact_degree_negatives(pos, st, kn, seed, pool),
    "N2-nearest": lambda pos, st, kn, seed, pool=None: sample_nearest_degree_negatives(pos, st, kn, seed, pool),
}


def endpoint_values(pairs: pd.DataFrame, scalar: pd.Series) -> np.ndarray:
    """Endpoint-occurrence values (each pair contributes both endpoints)."""
    return np.concatenate([pairs["protein_a"].map(scalar).to_numpy(float), pairs["protein_b"].map(scalar).to_numpy(float)])


def balance_row(pos_vals: np.ndarray, neg_vals: np.ndarray) -> dict:
    na, nb = len(pos_vals), len(neg_vals)
    pooled = np.sqrt(((na - 1) * pos_vals.var(ddof=1) + (nb - 1) * neg_vals.var(ddof=1)) / (na + nb - 2))
    return {"positive_mean": pos_vals.mean(), "negative_mean": neg_vals.mean(),
            "positive_median": np.median(pos_vals), "negative_median": np.median(neg_vals),
            "smd": (pos_vals.mean() - neg_vals.mean()) / pooled if pooled > 0 else 0.0,
            "ks_statistic": stats.ks_2samp(pos_vals, neg_vals).statistic,
            "wasserstein": stats.wasserstein_distance(pos_vals, neg_vals)}


# =============================================================================
# 5. Protein representations and symmetric pair features
# =============================================================================

CTD_ATTRIBUTES = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "normalized_vdw_volume": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary_structure": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent_accessibility": ("ALFCGIVW", "RKQEND", "MSPTHY"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
}
_CTD_INDEX = {attr: {aa: g for g, grp in enumerate(groups, 1) for aa in grp} for attr, groups in CTD_ATTRIBUTES.items()}


def compute_aac(seq: str) -> np.ndarray:
    """20-dim amino-acid composition."""
    return np.array([seq.count(aa) / len(seq) for aa in AMINO_ACIDS], dtype=np.float64)


def compute_ctd(seq: str) -> np.ndarray:
    """147-dim composition/transition/distribution descriptors (7 attributes x 21)."""
    n = len(seq)
    feats = []
    for attr in CTD_ATTRIBUTES:
        coded = np.array([_CTD_INDEX[attr][aa] for aa in seq])
        feats.extend([(coded == g).sum() / n for g in (1, 2, 3)])
        if n < 2:
            feats.extend([0.0, 0.0, 0.0])
        else:
            a, b = coded[:-1], coded[1:]
            for x, y in ((1, 2), (1, 3), (2, 3)):
                feats.append((((a == x) & (b == y)) | ((a == y) & (b == x))).sum() / (n - 1))
        for g in (1, 2, 3):
            pos = np.where(coded == g)[0]
            if len(pos) == 0:
                feats.extend([0.0] * 5)
                continue
            c = len(pos)
            marks = [pos[0], pos[max(int(c * 0.25) - 1, 0)], pos[max(int(c * 0.50) - 1, 0)],
                     pos[max(int(c * 0.75) - 1, 0)], pos[-1]]
            feats.extend([(p + 1) / n for p in marks])
    return np.array(feats, dtype=np.float64)


def aac_ctd_features(proteins: pd.DataFrame) -> pd.DataFrame:
    """AAC+CTD (167 dims). Non-standard residues are removed before computing."""
    rows = []
    for seq in proteins["sequence"]:
        clean = "".join(c for c in seq if c in AMINO_ACIDS) or "A"
        rows.append(np.concatenate([compute_aac(clean), compute_ctd(clean)]))
    out = pd.DataFrame(np.vstack(rows), columns=[f"aacctd_{i}" for i in range(167)])
    out.insert(0, "protein_id", proteins["protein_id"].to_numpy())
    return out


def pair_matrix(pairs: pd.DataFrame, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric fusion [hA+hB, |hA-hB|, hA*hB] so f(A,B) == f(B,A)."""
    table = feats.set_index("protein_id")
    ha = table.loc[pairs["protein_a"]].to_numpy(np.float64)
    hb = table.loc[pairs["protein_b"]].to_numpy(np.float64)
    return np.concatenate([ha + hb, np.abs(ha - hb), ha * hb], axis=1), pairs["label"].to_numpy()


def scalar_pair_matrix(pairs: pd.DataFrame, scalar: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Degree-only or length-only control: sum, |diff|, min, max of endpoint values."""
    a = pairs["protein_a"].map(scalar).to_numpy(float)
    b = pairs["protein_b"].map(scalar).to_numpy(float)
    return np.column_stack([a + b, np.abs(a - b), np.minimum(a, b), np.maximum(a, b)]), pairs["label"].to_numpy()


# =============================================================================
# 6. Models and metrics
# =============================================================================

def build_model(name: str, seed: int, control: bool = False):
    if name == "lr":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "rf":
        if control:
            return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=seed)
    raise ValueError(f"Unknown model {name!r}; choose from lr, rf, mlp, xgb")


MODEL_LABELS = {"lr": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost", "mlp": "MLP"}


def ece_equal_width(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1], right=True), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """ROC-AUC, PR-AUC (average precision) and fixed-threshold metrics (never tuned on test)."""
    if len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "mcc": np.nan, "n_test": len(y)}
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "mcc": matthews_corrcoef(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0, "brier": brier_score_loss(y, p),
            "ece_10bin": ece_equal_width(y, p), "positive_prevalence": float(y.mean()), "n_test": int(len(y))}


def fit_predict(model_name, x_train, y_train, x_test, seed, control=False) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        raise ValueError("training data contain a single class")
    scaler = StandardScaler().fit(x_train)
    model = build_model(model_name, seed, control)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(scaler.transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_test))[:, 1]


# =============================================================================
# 7. Graph-aware uncertainty
# =============================================================================

def component_groups(test_pairs: pd.DataFrame) -> list[np.ndarray]:
    """Row indices grouped by connected component of the test-pair graph."""
    comps = connected_components(test_pairs)
    node_to_comp = {n: i for i, comp in enumerate(comps) for n in comp}
    comp_of_row = test_pairs["protein_a"].map(node_to_comp).to_numpy()
    return [np.where(comp_of_row == c)[0] for c in range(len(comps))]


def component_bootstrap_ci(y, p, groups, n_boot: int, seed: int, null: float = 0.5) -> dict:
    """Percentile bootstrap resampling whole components with replacement."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    vals = np.array(vals)
    if len(vals) == 0:
        return {"boot_low": np.nan, "boot_high": np.nan, "boot_p": np.nan}
    return {"boot_low": np.percentile(vals, 2.5), "boot_high": np.percentile(vals, 97.5),
            "boot_p": min(1.0, 2 * min(np.mean(vals <= null), np.mean(vals >= null)))}


def component_jackknife_ci(y, p, groups, null: float = 0.5) -> dict:
    """Delete-one-component jackknife with a t(G_eff - 1) reference distribution."""
    point = roc_auc_score(y, p)
    all_idx = np.arange(len(y))
    loo = []
    for rows in groups:
        keep = np.setdiff1d(all_idx, rows, assume_unique=True)
        if len(np.unique(y[keep])) == 2:
            loo.append(roc_auc_score(y[keep], p[keep]))
    loo = np.array(loo)
    g = len(loo)
    if g < 2:
        return {"jack_low": np.nan, "jack_high": np.nan, "jack_p": np.nan, "components_used": g}
    se = np.sqrt((g - 1) / g * np.sum((loo - loo.mean()) ** 2))
    t = stats.t.ppf(0.975, df=g - 1)
    pval = (0.0 if point != null else 1.0) if se == 0 else 2 * stats.t.sf(abs((point - null) / se), df=g - 1)
    return {"jack_low": point - t * se, "jack_high": point + t * se, "jack_p": pval, "components_used": g}


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m:
        idx = np.where(ok)[0][np.argsort(p[ok])]
        adj = np.minimum.accumulate((p[idx] * m / np.arange(1, m + 1))[::-1])[::-1]
        out[idx] = np.clip(adj, 0, 1)
    return out.tolist()


# =============================================================================
# 8. Figures
# =============================================================================

def _style(ax, ylabel=None, chance=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(NEUTRAL)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if chance:
        ax.axhline(0.5, color=NEUTRAL, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1], 0.5, " chance", va="center", ha="left", fontsize=7, color=INK_2)


def plot_dataset(prot_stats: pd.DataFrame, report: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    c = report["canonical"]
    axes[0].bar(["Positive", "Negative (N0)"], [c["n_positive"], c["n_negative"]], color=[PALETTE[0], PALETTE[1]], width=0.6)
    for i, v in enumerate([c["n_positive"], c["n_negative"]]):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK)
    axes[0].set_title("A  Pairs after canonicalization", loc="left", fontsize=9, color=INK)
    _style(axes[0], "Pairs")
    lengths = prot_stats["length"].to_numpy()
    axes[1].hist(lengths, bins=np.logspace(np.log10(max(lengths.min(), 1)), np.log10(lengths.max() + 1), 40), color=PALETTE[0])
    axes[1].set_xscale("log")
    axes[1].axvline(PLM_MAX_RESIDUES, color=INK_2, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Sequence length (residues, log scale)", fontsize=8, color=INK_2)
    axes[1].set_title("B  Sequence length", loc="left", fontsize=9, color=INK)
    _style(axes[1], "Proteins")
    deg = prot_stats["positive_degree"].to_numpy()
    axes[2].hist(deg, bins=np.arange(deg.max() + 2) - 0.5, color=PALETTE[0])
    axes[2].set_xlabel("Positive-network degree", fontsize=8, color=INK_2)
    axes[2].set_title("C  Positive-network degree", loc="left", fontsize=9, color=INK)
    _style(axes[2], "Proteins")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_degree_balance(pos_vals: np.ndarray, neg_vals: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    def ecdf(v):
        x = np.sort(v)
        return x, np.arange(1, len(x) + 1) / len(x)
    x, yv = ecdf(pos_vals)
    ax.step(x, yv, where="post", color=INK, linewidth=2, label="Positive endpoints")
    for i, (name, vals) in enumerate(neg_vals.items()):
        x, yv = ecdf(vals)
        ax.step(x, yv, where="post", color=PALETTE[i % len(PALETTE)], linewidth=1.6, label=NEG_LABELS.get(name, name))
    ax.set_xlabel("Endpoint positive-network degree", fontsize=9, color=INK_2)
    ax.set_title("Endpoint-degree distribution by negative set (ECDF)", loc="left", fontsize=10, color=INK)
    _style(ax, "Cumulative fraction of endpoints")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_novelty(df: pd.DataFrame, path: Path):
    reps = list(df["representation"].unique())
    fig, axes = plt.subplots(1, len(reps), figsize=(5.2 * len(reps), 3.8), squeeze=False, layout="constrained")
    order = [r for r in ["R0", "R1", "R2", "R3", "R3-C"] if r in set(df["regime"])]
    for ax, rep in zip(axes[0], reps):
        sub = df[df["representation"] == rep]
        for i, model in enumerate(sub["model"].unique()):
            m = sub[sub["model"] == model].set_index("regime").reindex(order)
            xs = np.arange(len(order))
            color = PALETTE[i % len(PALETTE)]
            matched = [k for k, r in enumerate(order) if r in ("R1", "R2", "R3")]
            ax.plot(xs[matched], m["roc_auc"].to_numpy()[matched], color=color, linewidth=2, marker="o", markersize=6,
                    label=MODEL_LABELS.get(model, model))
            other = [k for k, r in enumerate(order) if r not in ("R1", "R2", "R3")]
            ax.plot(xs[other], m["roc_auc"].to_numpy()[other], color=color, linestyle="none", marker="D", markersize=6)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([REGIME_LABELS[r].replace(" ", "\n", 1) for r in order], fontsize=8)
        ax.set_title(f"Protein-novelty curve ({rep})", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Lines: matched R1-R3 strata (shared training set). Diamonds: independently sampled regimes.",
                      fontsize=7, color=INK_2)
        _style(ax, "ROC-AUC", chance=True)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _grouped_bars(ax, df, group_col, series_col, value_col, group_order, series_order, series_labels):
    width = 0.8 / len(series_order)
    for i, s in enumerate(series_order):
        vals = [df[(df[group_col] == g) & (df[series_col] == s)][value_col].mean() for g in group_order]
        xs = np.arange(len(group_order)) + (i - (len(series_order) - 1) / 2) * width
        ax.bar(xs, vals, width * 0.92, color=PALETTE[i % len(PALETTE)], label=series_labels.get(s, s))
        for x, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(range(len(group_order)))


def plot_topology(df: pd.DataFrame, path: Path):
    sub = df[df["model"] == "lr"]
    evals = [n for n in NEG_LABELS if n in set(sub["evaluated_on"])]
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    _grouped_bars(ax, sub, "evaluated_on", "feature", "roc_auc", evals, ["degree", "length"],
                  {"degree": "Degree-only", "length": "Length-only"})
    ax.set_xticklabels([NEG_TICKS[e] for e in evals], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_title("Topology/length-only controls: trained on N0, test negatives replaced", loc="left", fontsize=10, color=INK)
    _style(ax, "ROC-AUC", chance=True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_revalidation(df: pd.DataFrame, path: Path):
    reps = list(df["representation"].unique())
    negs = [n for n in ["N0", "N2", "N5"] if n in set(df["negative_set"])]
    fig, axes = plt.subplots(1, len(reps), figsize=(5.0 * len(reps), 3.8), squeeze=False, layout="constrained")
    for ax, rep in zip(axes[0], reps):
        sub = df[df["representation"] == rep]
        _grouped_bars(ax, sub, "regime", "negative_set", "roc_auc", ["R0", "R3"], negs, NEG_LABELS)
        ax.set_xticklabels([REGIME_LABELS["R0"], REGIME_LABELS["R3"]], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Consistent-negative revalidation ({rep}, MLP)", loc="left", fontsize=10, color=INK)
        _style(ax, "ROC-AUC", chance=True)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_uncertainty(df: pd.DataFrame, path: Path):
    """Point estimate with component-jackknife (thick) and component-bootstrap
    (thin) 95% intervals, clipped to the ROC-AUC range [0, 1]."""
    df = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(df) + 1.6), layout="constrained")
    ys = np.arange(len(df))[::-1].astype(float)
    for y, (_, r) in zip(ys, df.iterrows()):
        color = PALETTE[["N0", "N2", "N5"].index(r["negative_set"])] if r["negative_set"] in ("N0", "N2", "N5") else INK
        for lo, hi, dy, lw in ((r["jack_low"], r["jack_high"], 0.0, 3), (r["boot_low"], r["boot_high"], -0.22, 1.2)):
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            ax.plot([max(lo, 0), min(hi, 1)], [y + dy, y + dy], color=color, linewidth=lw, solid_capstyle="butt")
            if lo < 0:
                ax.plot(0, y + dy, marker="<", color=color, markersize=5)
            if hi > 1:
                ax.plot(1, y + dy, marker=">", color=color, markersize=5)
        ax.plot(r["roc_auc"], y, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.text(1.05, y, f"{r['roc_auc']:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.regime} / {r.negative_set} / {r.representation}\n(G = {r.n_components} components, n = {r.n_test})"
                        for r in df.itertuples()], fontsize=8)
    ax.set_xlim(-0.03, 1.13)
    ax.set_ylim(ys.min() - 0.6, ys.max() + 0.6)
    ax.axvline(0.5, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(NEUTRAL)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlabel("ROC-AUC (value at right)\nThick: jackknife 95% CI.  Thin: bootstrap 95% CI.  Arrow: CI truncated.",
                  fontsize=8, color=INK_2)
    ax.set_title("Graph-aware uncertainty on the both-unseen (R3) test graph", loc="left", fontsize=10, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_summary(novelty, topology, revalidation, balance, path: Path):
    """One-page dashboard of the four central diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout="constrained")
    ax = axes[0, 0]
    rep0 = novelty["representation"].iloc[0]
    sub = novelty[(novelty["representation"] == rep0) & (novelty["regime"].isin(["R1", "R2", "R3"]))]
    for i, model in enumerate(sub["model"].unique()):
        m = sub[sub["model"] == model].set_index("regime").reindex(["R1", "R2", "R3"])
        ax.plot(range(3), m["roc_auc"], marker="o", color=PALETTE[i], linewidth=2, label=MODEL_LABELS.get(model, model))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["R1\nseen-seen", "R2\none-unseen", "R3\nboth-unseen"], fontsize=8)
    ax.set_title(f"A  Protein-novelty curve ({rep0}, N0)", loc="left", fontsize=10, color=INK)
    _style(ax, "ROC-AUC", chance=True)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    b = balance.set_index("negative_set")
    names = [n for n in NEG_LABELS if n in b.index]
    ax.bar(range(len(names)), b.loc[names, "smd"], color=[PALETTE[i % len(PALETTE)] for i in range(len(names))], width=0.6)
    for i, n in enumerate(names):
        v = b.loc[n, "smd"]
        ax.text(i, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=INK_2)
    ax.axhline(0.1, color=NEUTRAL, linestyle=":", linewidth=1)
    ax.axhline(0, color=NEUTRAL, linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([NEG_TICKS[n] for n in names], fontsize=7)
    ax.set_title("B  Endpoint-degree imbalance (SMD, positive - negative)", loc="left", fontsize=10, color=INK)
    _style(ax, "Standardized mean difference")

    ax = axes[1, 0]
    t = topology[(topology["model"] == "lr") & (topology["feature"] == "degree")].set_index("evaluated_on")
    names = [n for n in NEG_LABELS if n in t.index]
    ax.bar(range(len(names)), t.loc[names, "roc_auc"], color=[PALETTE[i % len(PALETTE)] for i in range(len(names))], width=0.6)
    for i, n in enumerate(names):
        ax.text(i, t.loc[n, "roc_auc"] + 0.01, f"{t.loc[n, 'roc_auc']:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([NEG_TICKS[n] for n in names], fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_title("C  Degree-only classifier (trained on N0)", loc="left", fontsize=10, color=INK)
    _style(ax, "ROC-AUC", chance=True)

    ax = axes[1, 1]
    rv = revalidation[revalidation["representation"] == revalidation["representation"].iloc[0]]
    negs = [n for n in ["N0", "N2", "N5"] if n in set(rv["negative_set"])]
    _grouped_bars(ax, rv, "regime", "negative_set", "roc_auc", ["R0", "R3"], negs, NEG_LABELS)
    ax.set_xticklabels(["R0 random pairs", "R3 both-unseen"], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"D  Sequence model, consistent negatives ({rv['representation'].iloc[0]})", loc="left", fontsize=10, color=INK)
    _style(ax, "ROC-AUC", chance=True)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================
# 9. Framework runner
# =============================================================================

def run_framework(proteins_path, pairs_path, out_dir, seed=42, models=("lr", "rf", "xgb", "mlp"),
                  embeddings=None, embeddings_dataset="mean", embeddings_name=None, esm2_model=None, clusters_path=None,
                  n_boot=2000, test_fraction=0.2, extra_negatives=True, audit_only=False) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    for sub in ("tables", "figures", "splits", "negatives", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- 1. audit --------------------------------------------------------------
    log("1/7 Auditing and canonicalizing the dataset")
    proteins_raw, pairs_raw = read_proteins(proteins_path), read_pairs(pairs_path)
    proteins, pairs, audit = audit_and_canonicalize(proteins_raw, pairs_raw)
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=float))
    pairs.to_csv(out / "canonical_pairs.tsv", sep="\t", index=False)
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)
    stats_tbl = compute_protein_stats(proteins, pos)
    stats_tbl.to_csv(out / "tables" / "protein_statistics.csv")
    plot_dataset(stats_tbl, audit, out / "figures" / "01_dataset_audit.png")
    log(f"    {audit['canonical']['n_pairs']:,} canonical pairs "
        f"({audit['canonical']['n_positive']:,} positive / {audit['canonical']['n_negative']:,} negative); "
        f"{audit['issues']['duplicate_pairs_removed']} duplicates removed")
    if audit_only:
        log(f"Audit written to {out / 'audit.json'}")
        return {"audit": audit}
    if len(pos) < 20:
        raise ValueError("At least 20 positive pairs are required to run the framework.")

    known = known_pair_set(pos)
    supplied_negatives = len(neg0) > 0
    if not supplied_negatives:
        log("    No label-0 rows supplied: N0 will be the random N1 set (no original negatives to audit)")

    # ---- 2. negative sets --------------------------------------------------------
    log("2/7 Constructing putative-negative sets and balance diagnostics")
    neg_sets = {"N1": sample_random_negatives(list(proteins["protein_id"]), known, len(pos), seed)}
    if not supplied_negatives:
        neg0 = neg_sets["N1"].copy()
        pairs = pd.concat([pos, neg0], ignore_index=True)
    neg_sets = {"N0": neg0, **neg_sets}
    for name in (["N2", "N3", "N5"] + (["N2-exact", "N2-nearest"] if extra_negatives else [])):
        neg_sets[name] = NEGATIVE_BUILDERS[name](pos, stats_tbl, known, seed)
    for name, df in neg_sets.items():
        df.to_csv(out / "negatives" / f"{name}.tsv", sep="\t", index=False)
    deg, length = stats_tbl["positive_degree"], stats_tbl["length"]
    pos_deg = endpoint_values(pos, deg)
    balance = []
    for name, df in neg_sets.items():
        row = {"negative_set": name, "n_negatives": len(df), "positives_without_negative": df.attrs.get("n_skipped", 0)}
        row.update(balance_row(pos_deg, endpoint_values(df, deg)))
        row["length_smd"] = balance_row(endpoint_values(pos, length), endpoint_values(df, length))["smd"]
        balance.append(row)
    balance = pd.DataFrame(balance)
    balance.to_csv(out / "tables" / "negative_set_balance.csv", index=False)
    plot_degree_balance(pos_deg, {k: endpoint_values(v, deg) for k, v in neg_sets.items() if k in ("N0", "N1", "N2", "N3", "N5")},
                        out / "figures" / "02_degree_balance.png")
    b0 = balance.set_index("negative_set")
    log(f"    Mean endpoint degree: positives {b0.loc['N0', 'positive_mean']:.2f} vs N0 {b0.loc['N0', 'negative_mean']:.2f} "
        f"(SMD {b0.loc['N0', 'smd']:.3f}); N2 SMD {b0.loc['N2', 'smd']:.3f}")

    # ---- 3. splits ----------------------------------------------------------------
    log("3/7 Building protein-novelty splits with leakage gates")
    clusters = read_clusters(clusters_path) if clusters_path else None
    splits = make_splits(pairs, proteins, seed, test_fraction, 0.2, clusters)
    id_to_group = dict(zip(proteins["protein_id"], proteins["sequence_group_id"]))
    split_rows = []
    for name, (tr, te) in splits.items():
        checks = leakage_checks(name, tr, te, id_to_group, clusters)
        (out / "splits" / name).mkdir(exist_ok=True)
        tr.to_csv(out / "splits" / name / "train.tsv", sep="\t", index=False)
        te.to_csv(out / "splits" / name / "test.tsv", sep="\t", index=False)
        split_rows.append({"regime": name, "train_pairs": len(tr), "test_pairs": len(te),
                           "train_positive": int(tr.label.sum()), "test_positive": int(te.label.sum()),
                           "train_proteins": len(_proteins_of(tr)), "test_proteins": len(_proteins_of(te)),
                           "test_components": len(connected_components(te)), **checks})
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(out / "tables" / "split_summary.csv", index=False)
    log("    " + ", ".join(f"{r.regime}: {r.train_pairs}/{r.test_pairs}" for r in split_df.itertuples()) + " (train/test pairs); all gates passed")

    # ---- 4. representations -----------------------------------------------------------
    log("4/7 Computing protein representations")
    reps = {"AAC+CTD": aac_ctd_features(proteins)}
    if embeddings:
        emb = read_embeddings(embeddings, embeddings_dataset)
        missing = set(proteins["protein_id"]) - set(emb["protein_id"])
        if missing:
            raise ValueError(f"Embedding file lacks {len(missing)} proteins, e.g. {sorted(missing)[:5]}")
        reps[embeddings_name or f"Embedding ({Path(embeddings).stem})"] = emb
    if esm2_model:
        log(f"    Embedding {len(proteins):,} proteins with {esm2_model} (frozen, mean pooling)")
        reps["ESM-2 mean"] = compute_esm2_embeddings(proteins, esm2_model)
    log(f"    Representations: {', '.join(f'{k} ({v.shape[1] - 1} dims)' for k, v in reps.items())}")

    available = []
    for m in models:
        if m == "xgb":
            try:
                import xgboost  # noqa: F401
            except ImportError:
                log("    xgboost not installed: skipping XGBoost")
                continue
        available.append(m)
    all_preds = []

    # ---- 5. protein-novelty curve -----------------------------------------------------
    log(f"5/7 Protein-novelty curve: {len(reps)} representation(s) x {len(available)} model(s) x {len(splits)} regimes")
    nov_rows = []
    for rep_name, feats in reps.items():
        for regime, (tr, te) in splits.items():
            if te.empty or te.label.nunique() < 2:
                log(f"    {regime}: test set lacks both classes, skipped")
                continue
            x_tr, y_tr = pair_matrix(tr, feats)
            x_te, y_te = pair_matrix(te, feats)
            for model in available:
                p = fit_predict(model, x_tr, y_tr, x_te, seed)
                nov_rows.append({"representation": rep_name, "regime": regime, "model": model,
                                 "train_pairs": len(tr), "test_pairs": len(te), **classification_metrics(y_te, p)})
                all_preds.append(te[["protein_a", "protein_b", "label"]].assign(
                    experiment="novelty_curve", representation=rep_name, regime=regime, negative_set="N0", model=model, probability=p))
    novelty = pd.DataFrame(nov_rows)
    novelty.to_csv(out / "tables" / "novelty_curve_metrics.csv", index=False)
    plot_novelty(novelty, out / "figures" / "03_novelty_curve.png")
    first = novelty[novelty["representation"] == "AAC+CTD"].pivot(index="model", columns="regime", values="roc_auc")
    for model in first.index:
        log("    AAC+CTD " + MODEL_LABELS[model] + ": " + ", ".join(f"{r} {first.loc[model, r]:.3f}" for r in first.columns))

    # ---- 6. topology/length-only controls ----------------------------------------------
    log("6/7 Degree-only / length-only controls and consistent-negative revalidation")
    pos_tr, pos_te = train_test_split(pos, test_size=test_fraction, random_state=seed)
    tr_sets, te_sets = {}, {}
    for name, df in neg_sets.items():
        if len(df) < 5:
            continue
        n_tr, n_te = train_test_split(df, test_size=test_fraction, random_state=seed)
        tr_sets[name] = pd.concat([pos_tr, n_tr], ignore_index=True)
        te_sets[name] = pd.concat([pos_te, n_te], ignore_index=True)
    topo_rows = []
    for feature, scalar in (("degree", deg), ("length", length)):
        for model in ("lr", "rf"):
            x_tr, y_tr = scalar_pair_matrix(tr_sets["N0"], scalar)
            scaler = StandardScaler().fit(x_tr)
            clf = build_model(model, seed, control=True).fit(scaler.transform(x_tr), y_tr)
            for name, te in te_sets.items():
                x_te, y_te = scalar_pair_matrix(te, scalar)
                p = clf.predict_proba(scaler.transform(x_te))[:, 1]
                topo_rows.append({"feature": feature, "model": model, "trained_on": "N0", "evaluated_on": name,
                                  **classification_metrics(y_te, p)})
    topology = pd.DataFrame(topo_rows)
    topology.to_csv(out / "tables" / "topology_controls.csv", index=False)
    plot_topology(topology, out / "figures" / "04_topology_controls.png")
    t = topology[(topology.feature == "degree") & (topology.model == "lr")].set_index("evaluated_on")["roc_auc"]
    log("    Degree-only LR ROC-AUC: " + ", ".join(f"{k} {v:.3f}" for k, v in t.items()))

    # ---- consistent-negative revalidation (R0 and R3 under N0/N2/N5) ---------------
    train_pool, test_pool = partition_protein_pool(proteins, test_fraction, seed)
    r3_tr, r3_te = splits["R3"]
    reval_sets = {}
    for name in ("N0", "N2", "N5"):
        if name == "N0":
            r0 = (tr_sets["N0"], te_sets["N0"])
            r3 = (r3_tr, r3_te)
        else:
            r0 = (tr_sets[name], te_sets[name])
            tr_pos, te_pos = r3_tr[r3_tr.label == 1], r3_te[r3_te.label == 1]
            ntr = NEGATIVE_BUILDERS[name](tr_pos.reset_index(drop=True), stats_tbl, known, seed, train_pool)
            nte = NEGATIVE_BUILDERS[name](te_pos.reset_index(drop=True), stats_tbl, known, seed, test_pool)
            r3 = (pd.concat([tr_pos, ntr], ignore_index=True), pd.concat([te_pos, nte], ignore_index=True))
            if not _proteins_of(r3[0]).isdisjoint(_proteins_of(r3[1])):
                raise RuntimeError(f"R3/{name}: protein disjointness violated after negative substitution")
        reval_sets[("R0", name)] = r0
        reval_sets[("R3", name)] = r3
    reval_rows = []
    for rep_name, feats in reps.items():
        for (regime, name), (tr, te) in reval_sets.items():
            if te.label.nunique() < 2 or tr.label.nunique() < 2:
                continue
            x_tr, y_tr = pair_matrix(tr, feats)
            x_te, y_te = pair_matrix(te, feats)
            p = fit_predict("mlp", x_tr, y_tr, x_te, seed)
            reval_rows.append({"representation": rep_name, "regime": regime, "negative_set": name, "model": "mlp",
                               "train_pairs": len(tr), "test_pairs": len(te), **classification_metrics(y_te, p)})
            all_preds.append(te[["protein_a", "protein_b", "label"]].assign(
                experiment="consistent_negative", representation=rep_name, regime=regime, negative_set=name, model="mlp", probability=p))
    revalidation = pd.DataFrame(reval_rows)
    revalidation.to_csv(out / "tables" / "consistent_negative_revalidation.csv", index=False)
    plot_revalidation(revalidation, out / "figures" / "05_consistent_negative_revalidation.png")
    for r in revalidation.itertuples():
        if r.representation == "AAC+CTD":
            log(f"    AAC+CTD MLP {r.regime}/{r.negative_set}: ROC-AUC {r.roc_auc:.3f}")

    # ---- 7. graph-aware uncertainty ---------------------------------------------------
    log(f"7/7 Component-aware uncertainty (bootstrap n={n_boot}, delete-one-component jackknife)")
    preds = pd.concat(all_preds, ignore_index=True)
    unc_rows = []
    sel = preds[(preds.experiment == "consistent_negative") & (preds.regime == "R3")]
    for (rep_name, name), g in sel.groupby(["representation", "negative_set"], sort=False):
        g = g.reset_index(drop=True)
        y, p = g["label"].to_numpy(), g["probability"].to_numpy()
        groups = component_groups(g)
        row = {"regime": "R3", "negative_set": name, "representation": rep_name, "model": "mlp",
               "n_test": len(g), "n_components": len(groups), "largest_component_pairs": max(len(x) for x in groups),
               "roc_auc": roc_auc_score(y, p)}
        row.update(component_bootstrap_ci(y, p, groups, n_boot, seed))
        row.update(component_jackknife_ci(y, p, groups))
        unc_rows.append(row)
    uncertainty = pd.DataFrame(unc_rows)
    uncertainty["boot_p_bh"] = benjamini_hochberg(uncertainty["boot_p"].tolist())
    uncertainty["jack_p_bh"] = benjamini_hochberg(uncertainty["jack_p"].tolist())
    uncertainty["above_chance_both_methods"] = (uncertainty.roc_auc > 0.5) & (uncertainty.boot_p_bh < 0.05) & (uncertainty.jack_p_bh < 0.05)
    uncertainty.to_csv(out / "tables" / "component_aware_uncertainty.csv", index=False)
    plot_uncertainty(uncertainty, out / "figures" / "06_component_uncertainty.png")

    preds.to_csv(out / "predictions" / "sample_level_predictions.csv.gz", index=False, compression="gzip")
    plot_summary(novelty, topology, revalidation, balance, out / "figures" / "00_framework_summary.png")

    results = {"audit": audit, "balance": balance, "splits": split_df, "novelty": novelty, "topology": topology,
               "revalidation": revalidation, "uncertainty": uncertainty}
    write_report(out, results, supplied_negatives, seed, time.time() - t0, list(reps), available)
    log(f"Done in {time.time() - t0:.0f} s. Report: {out / 'REPORT.md'}")
    return results


# =============================================================================
# 10. Report
# =============================================================================

def _md_table(df: pd.DataFrame, cols: list[str], fmt: str = "{:.3f}") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(out: Path, res: dict, supplied_negatives: bool, seed: int, seconds: float, reps: list, models: list):
    a, bal, nov, topo, rv, unc = res["audit"], res["balance"], res["novelty"], res["topology"], res["revalidation"], res["uncertainty"]
    b = bal.set_index("negative_set")
    t = topo[(topo.feature == "degree") & (topo.model == "lr")].set_index("evaluated_on")["roc_auc"]
    rep0 = reps[0]
    nv = nov[(nov.representation == rep0)].pivot(index="model", columns="regime", values="roc_auc")
    rvp = rv[rv.representation == rep0].set_index(["regime", "negative_set"])["roc_auc"]

    flags = []
    if supplied_negatives and b.loc["N0", "smd"] > 0.25:
        flags.append(f"**Degree imbalance in supplied negatives.** Positive endpoints have mean degree "
                     f"{b.loc['N0', 'positive_mean']:.2f} versus {b.loc['N0', 'negative_mean']:.2f} for supplied negatives "
                     f"(SMD {b.loc['N0', 'smd']:.2f}).")
    if "N0" in t and t["N0"] > 0.75:
        flags.append(f"**Degree alone separates the supplied labels** (degree-only ROC-AUC {t['N0']:.3f}); "
                     f"it falls to {t.get('N2', np.nan):.3f} on degree-matched N2 negatives.")
    if "mlp" in nv.index and {"R1", "R3"} <= set(nv.columns):
        drop = nv.loc["mlp", "R1"] - nv.loc["mlp", "R3"]
        if drop > 0.03:
            flags.append(f"**Discrimination falls with protein novelty** (MLP ROC-AUC {nv.loc['mlp', 'R1']:.3f} at R1 "
                         f"to {nv.loc['mlp', 'R3']:.3f} at R3).")
    if ("R3", "N2") in rvp.index:
        v = rvp[("R3", "N2")]
        verdict = "close to chance" if abs(v - 0.5) < 0.05 else "above chance" if v > 0.55 else "below chance" if v < 0.45 else "modest"
        flags.append(f"**Consistent degree-matched negatives at R3: sequence-model point estimate {verdict}** "
                     f"(ROC-AUC {v:.3f}; N0 {rvp.get(('R3', 'N0'), np.nan):.3f}). See Section 7 for component-aware intervals.")
    if len(unc):
        flags.append(f"**Few independent test units:** R3 test graphs contain {unc.n_components.min()}-{unc.n_components.max()} "
                     f"connected components; interpret intervals and tests accordingly.")

    text = f"""# PPI benchmark-control report

Generated by `ppi_benchmark_controls.py` v{__version__} (seed {seed}, {seconds:.0f} s).
Representations: {', '.join(reps)}. Models: {', '.join(MODEL_LABELS[m] for m in models)}.

## Key diagnostics

{chr(10).join('- ' + f for f in flags) if flags else '- No diagnostic crossed its reporting threshold.'}

These are descriptive diagnostics on one dataset. Constructed negatives are putative
(unobserved) pairs, not experimentally established non-interactions; full-graph degree
is a retrospective audit feature, not a prospective input.

![Summary](figures/00_framework_summary.png)

## 1. Dataset audit

| Quantity | Value |
|---|---|
| Input proteins / pairs | {a['input']['n_proteins']:,} / {a['input']['n_pairs']:,} |
| Canonical pairs (positive / negative) | {a['canonical']['n_pairs']:,} ({a['canonical']['n_positive']:,} / {a['canonical']['n_negative']:,}) |
| Duplicate pairs removed (exact + reversed) | {a['issues']['duplicate_pairs_removed']:,} |
| Self-pairs / conflicting labels removed | {a['issues']['self_pairs_removed']} / {a['issues']['conflicting_label_pairs_removed']} |
| Identical-sequence groups (size > 1) | {a['canonical']['identical_sequence_groups_size_gt1']} |
| Proteins with zero positive degree | {a['canonical']['proteins_with_zero_positive_degree']:,} |
| Sequences longer than {PLM_MAX_RESIDUES} residues | {a['canonical'][f'sequences_longer_than_{PLM_MAX_RESIDUES}']:,} |
| Positive graph: nodes / edges / components / largest | {a['positive_graph'].get('nodes', 0):,} / {a['positive_graph'].get('edges', 0):,} / {a['positive_graph'].get('connected_components', 0)} / {a['positive_graph'].get('largest_component', 0):,} |

![Dataset](figures/01_dataset_audit.png)

## 2. Negative-set balance

Endpoint-occurrence statistics (each pair contributes two endpoints). SMD is positive minus negative.

{_md_table(bal, ['negative_set', 'n_negatives', 'positives_without_negative', 'positive_mean', 'negative_mean', 'smd', 'ks_statistic', 'wasserstein', 'length_smd'])}

![Degree balance](figures/02_degree_balance.png)

## 3. Splits and leakage gates

R1-R3 share one training set (matched protein-novelty curve); R0 is an independent random pair split.

{_md_table(res['splits'], ['regime', 'train_pairs', 'test_pairs', 'train_proteins', 'test_proteins', 'protein_overlap', 'reversed_pair_overlap', 'test_components', 'passed'])}

## 4. Protein-novelty curve (original negatives)

{_md_table(nov, ['representation', 'regime', 'model', 'test_pairs', 'roc_auc', 'pr_auc', 'mcc'])}

![Novelty](figures/03_novelty_curve.png)

## 5. Topology-only and length-only controls

Trained on N0; the test negatives are replaced by each alternative set.

{_md_table(topo[topo.model == 'lr'], ['feature', 'evaluated_on', 'roc_auc', 'pr_auc', 'mcc'])}

![Topology](figures/04_topology_controls.png)

## 6. Consistent-negative revalidation (MLP)

The same negative definition is used for training and testing. R3 negatives are sampled
within the train or test protein pool, so protein disjointness is preserved.

{_md_table(rv, ['representation', 'regime', 'negative_set', 'train_pairs', 'test_pairs', 'roc_auc', 'pr_auc', 'mcc'])}

![Revalidation](figures/05_consistent_negative_revalidation.png)

## 7. Component-aware uncertainty (R3)

Test pairs are grouped into connected components of the test-pair graph; whole components
are resampled (percentile bootstrap) or deleted one at a time (jackknife, t reference).
P values test ROC-AUC = 0.5 and are Benjamini-Hochberg adjusted within each method.

{_md_table(unc, ['representation', 'negative_set', 'n_test', 'n_components', 'roc_auc', 'boot_low', 'boot_high', 'jack_low', 'jack_high', 'boot_p_bh', 'jack_p_bh'])}

![Uncertainty](figures/06_component_uncertainty.png)

## Output files

- `audit.json`, `canonical_pairs.tsv`
- `splits/<regime>/{{train,test}}.tsv`, `negatives/<set>.tsv`
- `tables/*.csv` (all numbers above), `predictions/sample_level_predictions.csv.gz`
- `figures/*.png`
"""
    (out / "REPORT.md").write_text(text)


# =============================================================================
# 11. Synthetic demonstration data
# =============================================================================

def make_synthetic_dataset(out_dir: Path, n_proteins: int = 600, n_positive: int = 1800, seed: int = 7):
    """Toy benchmark with two properties the framework should detect:

    * a genuine, sequence-encoded interaction rule - proteins belong to hidden
      types; basic (K/R-rich) proteins preferentially bind acidic (D/E-rich)
      ones - so sequence models retain signal after degree matching; and
    * a planted construction bias - the supplied negatives are drawn mainly from
      low-degree proteins, so a degree-only classifier separates them easily.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = np.ones(20) / 20
    comp = {0: base.copy(), 1: base.copy(), 2: base.copy()}
    for aa in "KR":
        comp[1][AMINO_ACIDS.index(aa)] *= 3.0
    for aa in "DE":
        comp[2][AMINO_ACIDS.index(aa)] *= 3.0
    comp = {k: v / v.sum() for k, v in comp.items()}
    types = rng.choice([0, 1, 2], size=n_proteins, p=[0.4, 0.3, 0.3])
    lengths = np.clip(rng.lognormal(np.log(380), 0.55, n_proteins).astype(int), 60, 2500)
    ids = [f"SYN{i:04d}" for i in range(n_proteins)]
    seqs = ["".join(rng.choice(list(AMINO_ACIDS), size=L, p=comp[t])) for t, L in zip(types, lengths)]
    proteins = pd.DataFrame({"protein_id": ids, "sequence": seqs})

    activity = rng.pareto(2.5, n_proteins) + 0.3          # heavy-tailed propensity -> degree heterogeneity
    affinity = np.ones((3, 3))
    affinity[1, 2] = affinity[2, 1] = 6.0                   # basic-acidic complementarity
    positives: set = set()
    w = activity / activity.sum()
    while len(positives) < n_positive:
        u = rng.choice(n_proteins, p=w)
        pv = activity * affinity[types[u], types]
        pv[u] = 0
        v = rng.choice(n_proteins, p=pv / pv.sum())
        positives.add((min(ids[u], ids[v]), max(ids[u], ids[v])))
    pos = pd.DataFrame(sorted(positives), columns=["protein_a", "protein_b"]).assign(label=1)

    degree = pd.concat([pos.protein_a, pos.protein_b]).value_counts().reindex(ids, fill_value=0).to_numpy()
    wn = 1.0 / (1.0 + degree) ** 1.5                         # biased negatives: low-degree endpoints
    wn /= wn.sum()
    negatives: set = set()
    while len(negatives) < n_positive:
        a, b = rng.choice(n_proteins, size=2, p=wn)
        key = (min(ids[a], ids[b]), max(ids[a], ids[b]))
        if a != b and key not in positives:
            negatives.add(key)
    neg = pd.DataFrame(sorted(negatives), columns=["protein_a", "protein_b"]).assign(label=0)
    pairs = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed)
    # present half of the rows in reversed order and add a few duplicates, to exercise the audit
    flip = rng.random(len(pairs)) < 0.5
    pairs.loc[flip, ["protein_a", "protein_b"]] = pairs.loc[flip, ["protein_b", "protein_a"]].values
    pairs = pd.concat([pairs, pairs.sample(12, random_state=seed)], ignore_index=True)

    proteins.to_csv(out_dir / "proteins.tsv", sep="\t", index=False)
    pairs.to_csv(out_dir / "pairs.tsv", sep="\t", index=False)
    pd.DataFrame({"protein_id": ids, "hidden_type": types}).to_csv(out_dir / "hidden_types.tsv", sep="\t", index=False)
    return out_dir / "proteins.tsv", out_dir / "pairs.tsv"


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Subcommands")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--out", required=True, help="output directory")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--models", default="lr,rf,xgb,mlp", help="comma list from lr,rf,xgb,mlp (default: all)")
        p.add_argument("--n-boot", type=int, default=2000, help="component-bootstrap replicates (default 2000)")
        p.add_argument("--quick", action="store_true", help="LR+MLP only, 200 bootstrap replicates, no exact/nearest N2")

    p_demo = sub.add_parser("demo", help="synthetic data with a planted degree bias")
    common(p_demo)
    p_run = sub.add_parser("run", help="run the framework on your data")
    common(p_run)
    for p in (p_run,):
        p.add_argument("--proteins", required=True, help="protein table (TSV/CSV: protein_id, sequence) or FASTA")
        p.add_argument("--pairs", required=True, help="pair table (TSV/CSV: protein_a, protein_b[, label])")
        p.add_argument("--clusters", help="optional protein_id -> sequence-cluster table (adds the R3-C split)")
        p.add_argument("--embeddings", help="optional per-protein embedding file (.csv/.tsv/.npz/.h5)")
        p.add_argument("--embeddings-dataset", default="mean", help="HDF5 dataset to read (default: mean)")
        p.add_argument("--embeddings-name", help='label for the embedding in tables/figures, e.g. "ESM-2 mean"')
        p.add_argument("--esm2", nargs="?", const="facebook/esm2_t12_35M_UR50D", default=None,
                       help="compute frozen ESM-2 mean embeddings (needs torch + transformers)")
    p_val = sub.add_parser("validate", help="audit a dataset only")
    p_val.add_argument("--proteins", required=True)
    p_val.add_argument("--pairs", required=True)
    p_val.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.command == "validate":
        run_framework(args.proteins, args.pairs, args.out, audit_only=True)
        return
    models = ("lr", "mlp") if args.quick else tuple(m.strip() for m in args.models.split(",") if m.strip())
    n_boot = min(args.n_boot, 200) if args.quick else args.n_boot
    if args.command == "demo":
        data_dir = Path(args.out) / "synthetic_input"
        prot, pairs = make_synthetic_dataset(data_dir)
        log(f"Synthetic dataset written to {data_dir} (hidden types in hidden_types.tsv)")
        run_framework(prot, pairs, args.out, seed=args.seed, models=models, n_boot=n_boot, extra_negatives=not args.quick)
    else:
        run_framework(args.proteins, args.pairs, args.out, seed=args.seed, models=models, embeddings=args.embeddings,
                      embeddings_dataset=args.embeddings_dataset, embeddings_name=args.embeddings_name, esm2_model=args.esm2, clusters_path=args.clusters,
                      n_boot=n_boot, extra_negatives=not args.quick)


if __name__ == "__main__":
    sys.exit(main())
