"""Leakage-resistant dataset splits (R0-R3).

R1/R2/R3 are derived from ONE partition of proteins into a train-pool and a
test-pool (partitioned at the sequence_group_id level, so identical-
sequence paralogs never end up split across the boundary), and share ONE
training set — pairs with both proteins in the train pool, minus a held-out
R1 test slice. This is the standard Park-Marcotte-style C1/C2/C3 design:
train once, evaluate at three increasing difficulty levels.

R0 is independent of the protein pool: a naive random split of all
canonical pairs, ignoring protein identity — the traditional-benchmark
reference point the other three splits are measured against.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def partition_protein_pool(
    proteins_with_groups: pd.DataFrame, test_fraction: float, seed: int,
    group_col: str = "sequence_group_id", sort_groups: bool = False,
) -> tuple[set[str], set[str]]:
    """Split proteins into (train_pool, test_pool) protein_id sets, partitioning
    at the `group_col` level so no group (identical-sequence group, or
    an MMseqs2 homology cluster) is split across the boundary."""
    group_ids = proteins_with_groups[group_col].unique()
    if sort_groups:
        # MMseqs2 output row order is not stable across builds/runs. Sorting is
        # therefore required for similarity-cluster splits, but remains opt-in
        # so the established R1–R3 sequence-group partition is unchanged.
        group_ids = np.asarray(sorted(group_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(group_ids)
    n_test = int(round(len(shuffled) * test_fraction))
    test_groups = set(shuffled[:n_test])
    train_groups = set(shuffled[n_test:])

    train_pool = set(proteins_with_groups.loc[proteins_with_groups[group_col].isin(train_groups), "protein_id"])
    test_pool = set(proteins_with_groups.loc[proteins_with_groups[group_col].isin(test_groups), "protein_id"])
    assert train_pool.isdisjoint(test_pool)
    return train_pool, test_pool


def make_cluster_disjoint_split(
    pairs: pd.DataFrame, protein_clusters: pd.DataFrame, test_fraction: float, seed: int, group_col: str = "cluster_id"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An R3-analog split at a given homology-cluster resolution.
    Only 'both train-pool' and 'both test-pool' pairs are kept (mixed pairs
    dropped) — a single strict both-unseen-and-dissimilar test point."""
    train_pool, test_pool = partition_protein_pool(
        protein_clusters, test_fraction, seed, group_col=group_col, sort_groups=True
    )
    category = _pool_membership(pairs, train_pool, test_pool)
    train = pairs[category == "both_train"].reset_index(drop=True)
    test = pairs[category == "both_test"].reset_index(drop=True)
    return train, test


def _pool_membership(pairs: pd.DataFrame, train_pool: set[str], test_pool: set[str]) -> pd.Series:
    a_train = pairs["protein_a"].isin(train_pool)
    b_train = pairs["protein_b"].isin(train_pool)
    a_test = pairs["protein_a"].isin(test_pool)
    b_test = pairs["protein_b"].isin(test_pool)
    both_train = a_train & b_train
    both_test = a_test & b_test
    mixed = (a_train & b_test) | (a_test & b_train)
    category = pd.Series("unassigned", index=pairs.index)
    category[both_train] = "both_train"
    category[both_test] = "both_test"
    category[mixed] = "mixed"
    assert (category != "unassigned").all(), "every protein must be in exactly one pool"
    return category


def make_r0_random_split(pairs: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R0: naive random pair split, ignoring protein identity. Traditional benchmark."""
    train, test = train_test_split(pairs, test_size=test_fraction, stratify=pairs["label"], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _degree_safe_holdout(both_train_pairs: pd.DataFrame, holdout_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select an R1 holdout from `both_train_pairs` such that every protein
    keeps at least one edge in the remaining (shared) training set.

    A naive random pair-level split can strip ALL training edges from a
    low-degree protein, which silently breaks the R1/R2 guarantee that every
    train-pool protein has "individually appeared somewhere in training" —
    caught by tests/test_splits.py. This greedy degree-safe removal fixes it:
    a candidate pair is only moved to the holdout if both its endpoints will
    still have degree >= 1 in the training set afterward.
    """
    degree = pd.concat([both_train_pairs["protein_a"], both_train_pairs["protein_b"]]).value_counts().to_dict()

    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(both_train_pairs.index)
    target_n = int(round(len(both_train_pairs) * holdout_fraction))

    holdout_idx = []
    for idx in shuffled_idx:
        if len(holdout_idx) >= target_n:
            break
        row = both_train_pairs.loc[idx]
        a, b = row["protein_a"], row["protein_b"]
        if degree[a] > 1 and degree[b] > 1:
            degree[a] -= 1
            degree[b] -= 1
            holdout_idx.append(idx)

    holdout_idx = set(holdout_idx)
    r1_test = both_train_pairs.loc[both_train_pairs.index.isin(holdout_idx)].reset_index(drop=True)
    shared_train = both_train_pairs.loc[~both_train_pairs.index.isin(holdout_idx)].reset_index(drop=True)
    return shared_train, r1_test


def make_r1_r2_r3_splits(
    pairs: pd.DataFrame,
    proteins_with_groups: pd.DataFrame,
    test_protein_pool_fraction: float,
    r1_holdout_fraction: float,
    seed: int,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    train_pool, test_pool = partition_protein_pool(proteins_with_groups, test_protein_pool_fraction, seed)
    category = _pool_membership(pairs, train_pool, test_pool)

    both_train_pairs = pairs[category == "both_train"]
    r2_test = pairs[category == "mixed"].reset_index(drop=True)
    r3_test = pairs[category == "both_test"].reset_index(drop=True)

    shared_train, r1_test = _degree_safe_holdout(both_train_pairs, r1_holdout_fraction, seed)

    # Every protein in both_train_pairs must retain >=1 edge in shared_train
    # (guaranteed by _degree_safe_holdout above).
    shared_train_proteins = set(shared_train["protein_a"]) | set(shared_train["protein_b"])
    both_train_proteins = set(both_train_pairs["protein_a"]) | set(both_train_pairs["protein_b"])
    assert both_train_proteins.issubset(shared_train_proteins), "degree-safe holdout invariant violated"

    # A train-pool protein whose ONLY pool-eligible edges are "mixed" ones (it
    # never co-occurs with another train-pool protein) has zero degree in
    # both_train_pairs and therefore can never appear in shared_train. Its R2
    # test pairs would then violate "exactly one member absent from training"
    # (the train-pool side wouldn't actually be seen) — drop those pairs
    # rather than silently violate the split's own definition.
    train_side_of_mixed = r2_test["protein_a"].where(r2_test["protein_a"].isin(train_pool), r2_test["protein_b"])
    r2_test = r2_test[train_side_of_mixed.isin(shared_train_proteins)].reset_index(drop=True)

    return {
        "seen_seen": (shared_train, r1_test),
        "one_unseen": (shared_train, r2_test),
        "both_unseen": (shared_train, r3_test),
    }
