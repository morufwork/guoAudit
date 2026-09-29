"""MMseqs2 sequence-homology clustering + cluster-disjoint splits.

For each identity threshold (50/40/30/20%): cluster all 2,497 protein
sequences with `mmseqs easy-cluster`, partition clusters into a train/test
pool (same mechanism as the split generation's sequence_group_id partition, generalized),
and keep only "both train-pool" / "both test-pool" pairs — the R3-analog
"both-unseen-and-dissimilar" test point needed for the generalization
degradation curve.

Requires tools/mmseqs/bin/mmseqs — run scripts/00b_install_mmseqs2.sh first.

Writes:
  data/processed/clusters/cluster_<threshold>.tsv   — protein_id -> cluster_id
  data/splits/homology_<threshold>/{train,test}.tsv
  results/sequence_cluster_summary.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.homology import load_cluster_assignment, mmseqs_version, run_mmseqs_cluster, write_fasta
from src.data.leakage_checks import LeakageError, run_leakage_checks
from src.data.splits import make_cluster_disjoint_split
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("cluster_sequences")


def main():
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    homology_config = load_config(REPO_ROOT / "configs" / "homology.yaml")
    seed = homology_config["seed"]

    mmseqs_binary = REPO_ROOT / homology_config["mmseqs_binary"]
    if not mmseqs_binary.exists():
        raise FileNotFoundError(
            f"{mmseqs_binary} not found. Run scripts/00b_install_mmseqs2.sh first."
        )

    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pairs = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv",
        sep="\t",
        header=None,
        names=data_config["pairs_columns"],
    )

    fasta_path = REPO_ROOT / "data" / "interim" / "proteins.fasta"
    write_fasta(proteins, fasta_path)
    logger.info(f"Wrote {fasta_path}: {len(proteins)} sequences")

    clusters_dir = REPO_ROOT / "data" / "processed" / "clusters"
    summary = {
        "mmseqs_version": mmseqs_version(mmseqs_binary),
        "clustering_scope": "all 2497 proteins, independently at each threshold, before train/test assignment",
        "assignment_unit": "MMseqs2 cluster",
        "coverage": homology_config["coverage"],
        "cov_mode": homology_config["cov_mode"],
        "cluster_mode": homology_config["cluster_mode"],
        "sensitivity": homology_config["sensitivity"],
        "seq_id_mode": homology_config["seq_id_mode"],
        "command_template": "mmseqs easy-cluster INPUT OUTPUT TMP --min-seq-id TAU -c 0.8 --cov-mode 0 --cluster-mode 0 -s 4.0 --seq-id-mode 0 -v 1",
        "test_cluster_pool_fraction": homology_config["test_cluster_pool_fraction"],
        "thresholds": {},
    }

    for threshold in homology_config["thresholds"]:
        threshold_pct = int(round(threshold * 100))
        threshold_key = str(threshold_pct)
        logger.info(f"--- Clustering at {threshold_pct}% sequence identity ---")

        out_prefix = clusters_dir / f"mmseqs_{threshold_pct}"
        cluster_tsv = run_mmseqs_cluster(
            fasta_path,
            out_prefix,
            min_seq_id=threshold,
            coverage=homology_config["coverage"],
            cov_mode=homology_config["cov_mode"],
            cluster_mode=homology_config["cluster_mode"],
            sensitivity=homology_config["sensitivity"],
            seq_id_mode=homology_config["seq_id_mode"],
            mmseqs_binary=mmseqs_binary,
        )
        protein_clusters = load_cluster_assignment(cluster_tsv)
        assert set(protein_clusters["protein_id"]) == set(proteins["protein_id"]), "cluster assignment missing proteins"

        cluster_composition_path = clusters_dir / f"cluster_{threshold_pct}.tsv"
        protein_clusters.to_csv(cluster_composition_path, sep="\t", index=False)

        n_clusters = protein_clusters["cluster_id"].nunique()
        logger.info(f"[{threshold_pct}%] {len(proteins)} proteins -> {n_clusters} clusters")

        train, test = make_cluster_disjoint_split(
            pairs, protein_clusters, homology_config["test_cluster_pool_fraction"], seed
        )

        id_to_cluster = dict(zip(protein_clusters["protein_id"], protein_clusters["cluster_id"]))
        split_name = f"homology_{threshold_pct}"
        try:
            check_results = run_leakage_checks(split_name, train, test, id_to_cluster, require_protein_disjoint=True, logger=logger)
        except LeakageError:
            logger.error(f"Split '{split_name}' FAILED leakage checks — aborting, no files written for this threshold.")
            raise

        out_dir = REPO_ROOT / "data" / "splits" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(out_dir / "train.tsv", sep="\t", index=False)
        test.to_csv(out_dir / "test.tsv", sep="\t", index=False)

        train_proteins = set(train["protein_a"]) | set(train["protein_b"])
        test_proteins = set(test["protein_a"]) | set(test["protein_b"])
        summary["thresholds"][threshold_key] = {
            "n_clusters": int(n_clusters),
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
            f"[{threshold_pct}%] train={len(train)} pairs / {len(train_proteins)} proteins, "
            f"test={len(test)} pairs / {len(test_proteins)} proteins, shared proteins={len(train_proteins & test_proteins)}"
        )

    save_json(summary, REPO_ROOT / "results" / "sequence_cluster_summary.json")
    logger.info("Wrote results/sequence_cluster_summary.json")

    write_manifest(REPO_ROOT / "results", config={**data_config, **homology_config}, seed=seed, extra={"script": "03_cluster_sequences.py"})
    logger.info("Homology clustering + splits complete. All thresholds passed leakage checks.")


if __name__ == "__main__":
    main()
