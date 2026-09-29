"""Generate and validate the R0-R3 leakage-resistant splits.

Reads data/processed/ppi_pairs_clean.tsv and proteins_with_groups.tsv, never the raw dataset. Writes each split explicitly to
data/splits/{random,seen_seen,one_unseen,both_unseen}/{train,test}.tsv.
Splits are never regenerated implicitly by downstream training scripts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.leakage_checks import LeakageError, run_leakage_checks
from src.data.splits import _pool_membership, make_r0_random_split, make_r1_r2_r3_splits, partition_protein_pool
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("generate_splits")

# split_name -> (output_dirname, requires strict protein/group disjointness)
SPLIT_SPECS = {
    "random": ("random", False),
    "seen_seen": ("seen_seen", False),
    "one_unseen": ("one_unseen", False),
    "both_unseen": ("both_unseen", True),
}


def save_split(name: str, train: pd.DataFrame, test: pd.DataFrame):
    out_dir = REPO_ROOT / "data" / "splits" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.tsv", sep="\t", index=False)
    test.to_csv(out_dir / "test.tsv", sep="\t", index=False)
    logger.info(f"Wrote {out_dir}/train.tsv ({len(train)} pairs) and test.tsv ({len(test)} pairs)")


def main():
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    split_config = load_config(REPO_ROOT / "configs" / "splits.yaml")
    seed = split_config["seed"]

    pairs = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv",
        sep="\t",
        header=None,
        names=data_config["pairs_columns"],
    )
    proteins_with_groups = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    id_to_group = dict(zip(proteins_with_groups["protein_id"], proteins_with_groups["sequence_group_id"]))

    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    splits["random"] = make_r0_random_split(pairs, split_config["r0_test_fraction"], seed)
    splits.update(
        make_r1_r2_r3_splits(
            pairs,
            proteins_with_groups,
            split_config["test_protein_pool_fraction"],
            split_config["r1_holdout_fraction"],
            seed,
        )
    )

    # Exclusions must be logged, not silently dropped. A train-pool
    # protein whose only pool-eligible edges are "mixed" ones never appears
    # in shared_train, so its R2 pairs are excluded (see splits.py docstring).
    train_pool, test_pool = partition_protein_pool(proteins_with_groups, split_config["test_protein_pool_fraction"], seed)
    raw_mixed_count = int((_pool_membership(pairs, train_pool, test_pool) == "mixed").sum())
    n_r2_excluded = raw_mixed_count - len(splits["one_unseen"][1])
    if n_r2_excluded:
        logger.warning(
            f"Excluded {n_r2_excluded} candidate R2 (one_unseen) pairs whose train-pool protein "
            f"never appears in shared_train (zero both-train-pool degree) — see split_summary.json['exclusions']"
        )

    summary = {"exclusions": {"one_unseen_pairs_excluded_orphan_train_protein": n_r2_excluded}}
    for split_key, (dirname, require_disjoint) in SPLIT_SPECS.items():
        train, test = splits[split_key]

        try:
            check_results = run_leakage_checks(dirname, train, test, id_to_group, require_disjoint, logger=logger)
        except LeakageError:
            logger.error(f"Split '{dirname}' FAILED leakage checks — aborting, no files written for this split.")
            raise

        save_split(dirname, train, test)

        train_proteins = set(train["protein_a"]) | set(train["protein_b"])
        test_proteins = set(test["protein_a"]) | set(test["protein_b"])
        summary[dirname] = {
            "n_train_pairs": int(len(train)),
            "n_test_pairs": int(len(test)),
            "train_label_balance": train["label"].value_counts().to_dict(),
            "test_label_balance": test["label"].value_counts().to_dict(),
            "n_train_proteins": len(train_proteins),
            "n_test_proteins": len(test_proteins),
            "n_shared_proteins": len(train_proteins & test_proteins),
            "leakage_checks": check_results,
        }
        logger.info(
            f"[{dirname}] train={len(train)} pairs / {len(train_proteins)} proteins, "
            f"test={len(test)} pairs / {len(test_proteins)} proteins, "
            f"shared proteins={len(train_proteins & test_proteins)}"
        )

    save_json(summary, REPO_ROOT / "results" / "split_summary.json")
    logger.info("Wrote results/split_summary.json")

    write_manifest(REPO_ROOT / "results", config={**data_config, **split_config}, seed=seed, extra={"script": "04_generate_splits.py"})
    logger.info("Split generation complete. All splits passed leakage checks.")


if __name__ == "__main__":
    main()
