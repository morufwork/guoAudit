"""Negative-sampling strategies.

Terminology: every pair produced here is a
*constructed unknown/unobserved pair used as a negative for training or
evaluation*, not a confirmed non-interaction. Only the benchmark's own
supplied negatives (N0) carry whatever evidentiary status the original
Guo et al. construction gave them (undocumented in the source data).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import partition_protein_pool


def compute_protein_stats(proteins: pd.DataFrame, positive_pairs: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """protein_id -> sequence length, positive-degree, and decile bins for both."""
    degree = pd.concat([positive_pairs["protein_a"], positive_pairs["protein_b"]]).value_counts()
    stats = proteins[["protein_id", "sequence"]].copy()
    stats["length"] = stats["sequence"].str.len()
    stats["positive_degree"] = stats["protein_id"].map(degree).fillna(0).astype(int)
    stats["length_decile"] = pd.qcut(stats["length"], n_deciles, labels=False, duplicates="drop")
    # rank-based decile handles the heavy tie mass at low degree (many proteins
    # share degree 0/1/2) better than qcut, which can collapse on ties.
    stats["degree_decile"] = pd.qcut(stats["positive_degree"].rank(method="first"), n_deciles, labels=False, duplicates="drop")
    return stats.drop(columns="sequence").set_index("protein_id")


def _known_pair_set(pairs: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(pairs["protein_a"], pairs["protein_b"])) | set(zip(pairs["protein_b"], pairs["protein_a"]))


def sample_random_negatives(protein_ids: list[str], known_pairs: set, n: int, seed: int) -> pd.DataFrame:
    """N1: fully random pairs, excluding known positives and self-pairs."""
    rng = np.random.default_rng(seed)
    negatives = set()
    protein_ids = np.array(protein_ids)
    while len(negatives) < n:
        batch = rng.choice(protein_ids, size=(n * 2, 2))
        for a, b in batch:
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in known_pairs or key in negatives:
                continue
            negatives.add(key)
            if len(negatives) >= n:
                break
    rows = list(negatives)[:n]
    return pd.DataFrame(rows, columns=["protein_a", "protein_b"]).assign(label=0)


def sample_matched_negatives(
    positive_pairs: pd.DataFrame, protein_stats: pd.DataFrame, known_pairs: set,
    match_cols: list[str], seed: int, candidate_pool: set[str] | None = None,
) -> pd.DataFrame:
    """N2/N3/N5: one-sided substitution. For each positive pair (u,v), replace
    one endpoint (chosen at random) with a decoy protein sharing the same
    decile bucket(s) in `match_cols` -- e.g. ['degree_decile'] for N2,
    ['length_decile'] for N3, both for N5 (hard negatives: resemble a real
    positive pair under simple sequence/network characteristics).

    `candidate_pool`, if given, restricts eligible decoys to that protein_id
    set -- required to keep a matched negative set consistent with a
    protein-disjoint split (e.g. R3): decoys for a train-pool pair must stay
    in the train pool, decoys for a test-pool pair must stay in the test pool,
    or the negative-set swap would silently reopen the leakage that the protein-disjoint split closed.
    """
    rng = np.random.default_rng(seed)
    stats_for_buckets = protein_stats.loc[protein_stats.index.isin(candidate_pool)] if candidate_pool is not None else protein_stats
    bucket_members = stats_for_buckets.groupby(match_cols).apply(lambda g: list(g.index), include_groups=False).to_dict()

    negatives = []
    seen = set()
    for _, row in positive_pairs.sample(frac=1.0, random_state=seed).iterrows():
        u, v = row["protein_a"], row["protein_b"]
        replace_u = rng.random() < 0.5
        fixed, to_replace = (v, u) if replace_u else (u, v)
        bucket_key = tuple(protein_stats.loc[to_replace, match_cols]) if len(match_cols) > 1 else protein_stats.loc[to_replace, match_cols[0]]
        candidates = bucket_members.get(bucket_key, [])

        decoy = None
        rng.shuffle(candidates := list(candidates))
        for cand in candidates:
            if cand == fixed or cand == to_replace:
                continue
            key = (cand, fixed) if cand < fixed else (fixed, cand)
            if key in known_pairs or key in seen:
                continue
            decoy = cand
            break
        if decoy is None:
            continue  # no valid decoy in this bucket -- skip rather than fabricate a match

        pair = (decoy, fixed) if decoy < fixed else (fixed, decoy)
        seen.add(pair)
        negatives.append(pair)

    return pd.DataFrame(negatives, columns=["protein_a", "protein_b"]).assign(label=0)


def sample_exact_degree_matched_negatives(
    positive_pairs: pd.DataFrame, protein_stats: pd.DataFrame, known_pairs: set,
    seed: int, candidate_pool: set[str] | None = None,
) -> pd.DataFrame:
    """N2-exact: matches the replaced endpoint's
    *exact* positive_degree value, rather than the coarser decile bucket
    `sample_matched_negatives` uses for N2 -- the concern is
    that a decile can span e.g. degree 1 and degree 3, or degree 10 and 15.
    Matching is exact where possible: a pair is skipped
    (never matched to the nearest-available degree instead) when no protein
    in the candidate pool shares the exact degree value, so this set is
    smaller than N2 but strictly free of decile coarseness. `df.attrs["n_skipped_no_exact_match"]` records how many positive
    pairs had no exact-degree decoy available."""
    rng = np.random.default_rng(seed)
    stats_for_buckets = protein_stats.loc[protein_stats.index.isin(candidate_pool)] if candidate_pool is not None else protein_stats
    bucket_members = stats_for_buckets.groupby("positive_degree").apply(lambda g: list(g.index), include_groups=False).to_dict()

    negatives = []
    seen = set()
    n_skipped = 0
    for _, row in positive_pairs.sample(frac=1.0, random_state=seed).iterrows():
        u, v = row["protein_a"], row["protein_b"]
        replace_u = rng.random() < 0.5
        fixed, to_replace = (v, u) if replace_u else (u, v)
        degree_val = protein_stats.loc[to_replace, "positive_degree"]
        candidates = list(bucket_members.get(degree_val, []))
        rng.shuffle(candidates)

        decoy = None
        for cand in candidates:
            if cand == fixed or cand == to_replace:
                continue
            key = (cand, fixed) if cand < fixed else (fixed, cand)
            if key in known_pairs or key in seen:
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
    df.attrs["n_skipped_no_exact_match"] = n_skipped
    return df


def sample_nearest_degree_matched_negatives(
    positive_pairs: pd.DataFrame, protein_stats: pd.DataFrame, known_pairs: set,
    seed: int, candidate_pool: set[str] | None = None, caliper: float | None = None,
) -> pd.DataFrame:
    """N2-nn: true nearest-neighbor matching on
    the raw positive_degree scalar -- always picks the candidate whose degree
    is numerically closest to the replaced endpoint's, rather than any
    co-bucketed protein regardless of within-bucket distance (decile) or
    requiring an exact tie (`sample_exact_degree_matched_negatives`). With a
    single scalar confounder, this is the discrete-substitution equivalent of
    nearest-neighbor propensity-score matching: a propensity score estimated
    from degree alone is a monotonic function of degree, so nearest-neighbor
    matching on the propensity score and on raw degree select the same
    decoys. `caliper`, if given, bounds the maximum allowed |degree
    difference| a match may have (a pair with no candidate inside the caliper
    is skipped); `caliper=None` always accepts the closest available protein,
    however far, as long as the candidate pool is non-empty.

    Within each exact-distance tier, candidates are visited in a fresh random
    order *per query* (not a single fixed array order reused across every
    query). An earlier version walked a single globally-sorted array with a
    deterministic lo/hi tie-break, which meant every query sharing a target
    degree tried the same candidate first -- on the multi-species replication
    dataset this let a handful of decoy
    proteins get reused in up to ~4% of all generated pairs (1,378/31,678),
    letting sequence-based models trivially learn decoy identity rather than
    anything about degree-matching quality. Per-query randomization spreads
    reuse across every candidate at a given distance instead of concentrating
    it on whichever protein happens to sort first."""
    rng = np.random.default_rng(seed)
    stats_for_pool = protein_stats.loc[protein_stats.index.isin(candidate_pool)] if candidate_pool is not None else protein_stats
    degree_groups = stats_for_pool.groupby("positive_degree").apply(lambda g: g.index.to_numpy(), include_groups=False)
    uniq_degrees = degree_groups.index.to_numpy()
    group_arrays = degree_groups.to_numpy()

    negatives = []
    seen = set()
    n_skipped = 0
    decoy_counts: dict[str, int] = {}
    for _, row in positive_pairs.sample(frac=1.0, random_state=seed).iterrows():
        u, v = row["protein_a"], row["protein_b"]
        replace_u = rng.random() < 0.5
        fixed, to_replace = (v, u) if replace_u else (u, v)
        target = protein_stats.loc[to_replace, "positive_degree"]

        dist = np.abs(uniq_degrees - target)
        decoy = None
        for gi in np.argsort(dist, kind="stable"):
            d = dist[gi]
            if caliper is not None and d > caliper:
                break
            group = group_arrays[gi]
            for j in rng.permutation(len(group)):
                cand = group[j]
                if cand == fixed or cand == to_replace:
                    continue
                key = (cand, fixed) if cand < fixed else (fixed, cand)
                if key in known_pairs or key in seen:
                    continue
                decoy = cand
                break
            if decoy is not None:
                break
        if decoy is None:
            n_skipped += 1
            continue

        pair = (decoy, fixed) if decoy < fixed else (fixed, decoy)
        seen.add(pair)
        negatives.append(pair)
        decoy_counts[decoy] = decoy_counts.get(decoy, 0) + 1

    if decoy_counts and negatives:
        max_decoy, max_count = max(decoy_counts.items(), key=lambda kv: kv[1])
        reuse_fraction = max_count / len(negatives)
        if reuse_fraction > 0.02:
            import warnings
            warnings.warn(
                f"sample_nearest_degree_matched_negatives: decoy protein {max_decoy!r} was reused in "
                f"{max_count}/{len(negatives)} ({reuse_fraction:.1%}) generated negatives, exceeding the "
                f"2% single-decoy reuse guard -- this can let sequence-based models learn decoy identity "
                f"rather than degree-matching quality; inspect the degree distribution's candidate density "
                f"before trusting downstream results built on this negative set.",
                stacklevel=2,
            )

    df = pd.DataFrame(negatives, columns=["protein_a", "protein_b"]).assign(label=0)
    df.attrs["n_skipped_no_neighbor"] = n_skipped
    df.attrs["max_decoy_reuse_count"] = max(decoy_counts.values()) if decoy_counts else 0
    df.attrs["max_decoy_reuse_fraction"] = (max(decoy_counts.values()) / len(negatives)) if decoy_counts and negatives else 0.0
    return df


def sample_homology_controlled_matched_negatives(
    positive_pairs: pd.DataFrame, protein_stats: pd.DataFrame, known_pairs: set,
    match_cols: list[str], seed: int, cluster_assignment: dict,
    candidate_pool: set[str] | None = None,
) -> pd.DataFrame:
    """N7: the same one-sided decile-matched
    substitution as `sample_matched_negatives`, with an added homology
    exclusion -- a candidate decoy sharing the fixed endpoint's sequence
    cluster (`cluster_assignment`, from Section 2.3's MMseqs2 clustering at a
    chosen identity threshold) is never accepted, however good a degree/
    length match it is. This addresses the concern that a
    degree-matched decoy could happen to be a close homolog, family member,
    or (by proxy, since this dataset has no complex/family annotation --
    Section 2.6's N4 exclusion) functionally related to the fixed protein,
    which could inflate or deflate sequence-based model performance for
    reasons unrelated to degree matching itself. `df.attrs
    ["n_skipped_homology_exhausted"]` counts positive pairs where every
    remaining candidate in the matched bucket was excluded specifically for
    sharing the fixed endpoint's cluster (as opposed to the bucket being
    empty or exhausted by other exclusions already tracked by
    `sample_matched_negatives`)."""
    rng = np.random.default_rng(seed)
    stats_for_buckets = protein_stats.loc[protein_stats.index.isin(candidate_pool)] if candidate_pool is not None else protein_stats
    bucket_members = stats_for_buckets.groupby(match_cols).apply(lambda g: list(g.index), include_groups=False).to_dict()

    negatives = []
    seen = set()
    n_skipped_homology_exhausted = 0
    for _, row in positive_pairs.sample(frac=1.0, random_state=seed).iterrows():
        u, v = row["protein_a"], row["protein_b"]
        replace_u = rng.random() < 0.5
        fixed, to_replace = (v, u) if replace_u else (u, v)
        bucket_key = tuple(protein_stats.loc[to_replace, match_cols]) if len(match_cols) > 1 else protein_stats.loc[to_replace, match_cols[0]]
        candidates = list(bucket_members.get(bucket_key, []))
        rng.shuffle(candidates)

        decoy = None
        saw_homology_exclusion = False
        fixed_cluster = cluster_assignment.get(fixed)
        for cand in candidates:
            if cand == fixed or cand == to_replace:
                continue
            if cluster_assignment.get(cand) == fixed_cluster:
                saw_homology_exclusion = True
                continue
            key = (cand, fixed) if cand < fixed else (fixed, cand)
            if key in known_pairs or key in seen:
                continue
            decoy = cand
            break
        if decoy is None:
            if saw_homology_exclusion:
                n_skipped_homology_exhausted += 1
            continue

        pair = (decoy, fixed) if decoy < fixed else (fixed, decoy)
        seen.add(pair)
        negatives.append(pair)

    df = pd.DataFrame(negatives, columns=["protein_a", "protein_b"]).assign(label=0)
    df.attrs["n_skipped_homology_exhausted"] = n_skipped_homology_exhausted
    return df


def sample_two_sided_matched_negatives(
    positive_pairs: pd.DataFrame, protein_stats: pd.DataFrame, known_pairs: set,
    match_cols: list[str], seed: int, candidate_pool: set[str] | None = None,
    max_attempts_per_pair: int = 500,
) -> pd.DataFrame:
    """N6: a two-sided negative construction,
    unlike `sample_matched_negatives` (N2/N3/N5)'s one-sided substitution.
    N2/N3/N5 always anchor one decoy endpoint at a real protein from the
    source positive pair (u,v) and substitute only the other -- so every N2
    decoy shares one true interactor with a real positive, a property N0's
    own construction is not known to have. This function instead draws BOTH
    decoy endpoints fresh from the protein pool: for each positive pair
    (u,v), it samples a wholly new pair (c,d), c drawn from u's decile
    bucket(s) and d from v's, so the *joint* bucket distribution matches the
    positive pair's without requiring either endpoint to be u, v, or any
    other real interactor. This isolates whether the one-sided-substitution
    mechanism itself (as opposed to degree matching) contributes to the
    N0-vs-N2 collapse: if N6 collapses the same way N2 does, the mechanism
    is not the explanation; degree is.

    `candidate_pool` behaves as in `sample_matched_negatives` (keeps decoys
    inside the correct train/test protein pool for a protein-disjoint
    split). Uses bounded random retries (`max_attempts_per_pair`) rather
    than an exhaustive scan, since the two-sided candidate space
    (|bucket(u)| x |bucket(v)|) is much larger than the one-sided case."""
    rng = np.random.default_rng(seed)
    stats_for_buckets = protein_stats.loc[protein_stats.index.isin(candidate_pool)] if candidate_pool is not None else protein_stats
    bucket_members = stats_for_buckets.groupby(match_cols).apply(lambda g: list(g.index), include_groups=False).to_dict()

    def _bucket_key(protein_id):
        return tuple(protein_stats.loc[protein_id, match_cols]) if len(match_cols) > 1 else protein_stats.loc[protein_id, match_cols[0]]

    negatives = []
    seen = set()
    for _, row in positive_pairs.sample(frac=1.0, random_state=seed).iterrows():
        u, v = row["protein_a"], row["protein_b"]
        pool_u = bucket_members.get(_bucket_key(u), [])
        pool_v = bucket_members.get(_bucket_key(v), [])
        if not pool_u or not pool_v:
            continue

        decoy_pair = None
        for _attempt in range(max_attempts_per_pair):
            c, d = rng.choice(pool_u), rng.choice(pool_v)
            if c == d:
                continue
            pair = (c, d) if c < d else (d, c)
            if pair in known_pairs or pair in seen:
                continue
            decoy_pair = pair
            break
        if decoy_pair is None:
            continue  # bucket exhausted within the attempt budget -- skip rather than fabricate

        seen.add(decoy_pair)
        negatives.append(decoy_pair)

    return pd.DataFrame(negatives, columns=["protein_a", "protein_b"]).assign(label=0)


def build_r3_n0_n2_datasets(
    protein_stats: pd.DataFrame, known_pairs: set, proteins: pd.DataFrame, repo_root: Path,
    seed: int, test_fraction: float = 0.20,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Shared by the architecture ablation (architecture ablation) and the statistical analysis (statistical
    analysis): the R3 (both-unseen) split under N0 (the split generation's saved split,
    unchanged) and N2 (degree-matched negatives respecting the same
    train/test protein-pool partition, so substituting negatives can't
    reopen the leakage that the protein-disjoint split closed)."""
    train_pool, test_pool = partition_protein_pool(proteins, test_fraction, seed)
    r3_train_raw = pd.read_csv(repo_root / "data" / "splits" / "both_unseen" / "train.tsv", sep="\t")
    r3_test_raw = pd.read_csv(repo_root / "data" / "splits" / "both_unseen" / "test.tsv", sep="\t")
    r3_train_pos = r3_train_raw[r3_train_raw["label"] == 1].reset_index(drop=True)
    r3_test_pos = r3_test_raw[r3_test_raw["label"] == 1].reset_index(drop=True)

    train_neg = sample_matched_negatives(r3_train_pos, protein_stats, known_pairs, ["degree_decile"], seed, candidate_pool=train_pool)
    test_neg = sample_matched_negatives(r3_test_pos, protein_stats, known_pairs, ["degree_decile"], seed, candidate_pool=test_pool)
    n2_train = pd.concat([r3_train_pos, train_neg], ignore_index=True)
    n2_test = pd.concat([r3_test_pos, test_neg], ignore_index=True)

    return {"n0": (r3_train_raw, r3_test_raw), "n2": (n2_train, n2_test)}


def build_r0_n0_n2_n5_datasets(
    pos: pd.DataFrame, neg0: pd.DataFrame, protein_stats: pd.DataFrame, known_pairs: set,
    seed: int, test_fraction: float = 0.20,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Shared by the negative-bias re-validation and the
    ProtT5 cross-check: the R0 (random) split under N0 (the benchmark's own
    negatives), N2 (degree-matched), and N5 (degree+length-matched hard
    negatives), used *consistently* (same negative set for train and test,
    not swapped as in the negative-sampling study's original A/B design)."""
    from sklearn.model_selection import train_test_split

    pos_train, pos_test = train_test_split(pos, test_size=test_fraction, random_state=seed)

    neg_generators = {
        "n0": lambda: neg0,
        "n2": lambda: sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], seed),
        "n5": lambda: sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile", "length_decile"], seed),
    }

    datasets = {}
    for name, gen in neg_generators.items():
        neg_df = gen()
        neg_train, neg_test = train_test_split(neg_df, test_size=test_fraction, random_state=seed)
        datasets[name] = (
            pd.concat([pos_train, neg_train], ignore_index=True),
            pd.concat([pos_test, neg_test], ignore_index=True),
        )
    return datasets


def build_r3_n0_n2_n5_datasets(
    protein_stats: pd.DataFrame, known_pairs: set, proteins: pd.DataFrame, repo_root: Path,
    seed: int, test_fraction: float = 0.20,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """R3 (both-unseen) counterpart of `build_r0_n0_n2_n5_datasets`, adding N5
    to `build_r3_n0_n2_datasets`'s N0/N2. N2/N5 decoys respect the same
    train/test protein-pool partition the split generation used, so substituting
    negatives can't reopen the leakage that the protein-disjoint split closed (asserted below)."""
    train_pool, test_pool = partition_protein_pool(proteins, test_fraction, seed)
    r3_train_raw = pd.read_csv(repo_root / "data" / "splits" / "both_unseen" / "train.tsv", sep="\t")
    r3_test_raw = pd.read_csv(repo_root / "data" / "splits" / "both_unseen" / "test.tsv", sep="\t")
    r3_train_pos = r3_train_raw[r3_train_raw["label"] == 1].reset_index(drop=True)
    r3_test_pos = r3_test_raw[r3_test_raw["label"] == 1].reset_index(drop=True)

    datasets = {"n0": (r3_train_raw, r3_test_raw)}
    for name, cols in [("n2", ["degree_decile"]), ("n5", ["degree_decile", "length_decile"])]:
        train_neg = sample_matched_negatives(r3_train_pos, protein_stats, known_pairs, cols, seed, candidate_pool=train_pool)
        test_neg = sample_matched_negatives(r3_test_pos, protein_stats, known_pairs, cols, seed, candidate_pool=test_pool)
        train_df = pd.concat([r3_train_pos, train_neg], ignore_index=True)
        test_df = pd.concat([r3_test_pos, test_neg], ignore_index=True)

        train_endpoints = set(train_df["protein_a"]) | set(train_df["protein_b"])
        test_endpoints = set(test_df["protein_a"]) | set(test_df["protein_b"])
        assert train_endpoints.isdisjoint(test_endpoints), f"R3/{name}: protein-disjointness violated after negative substitution"

        datasets[name] = (train_df, test_df)
    return datasets
