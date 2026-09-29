"""MMseqs2 sequence-homology clustering.

Wraps `mmseqs easy-cluster` and parses its cluster assignment into a
protein_id -> cluster_id table, usable anywhere the split generation's sequence_group_id
was used (train_pool/test_pool partitioning, leakage checks).
"""
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def write_fasta(proteins: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for protein_id, sequence in zip(proteins["protein_id"], proteins["sequence"]):
            f.write(f">{protein_id}\n{sequence}\n")


def run_mmseqs_cluster(
    fasta_path: str | Path,
    out_prefix: str | Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    sensitivity: float,
    seq_id_mode: int,
    mmseqs_binary: str | Path,
) -> Path:
    """Runs `mmseqs easy-cluster`, returns the path to `<out_prefix>_cluster.tsv`."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            str(mmseqs_binary), "easy-cluster",
            str(fasta_path), str(out_prefix), tmp_dir,
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--cov-mode", str(cov_mode),
            "--cluster-mode", str(cluster_mode),
            "-s", str(sensitivity),
            "--seq-id-mode", str(seq_id_mode),
            "-v", "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mmseqs easy-cluster failed (min_seq_id={min_seq_id}):\n{result.stdout}\n{result.stderr}")

    cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
    assert cluster_tsv.exists(), f"expected {cluster_tsv} not produced"
    return cluster_tsv


def mmseqs_version(mmseqs_binary: str | Path) -> str:
    """Return the exact MMseqs2 build identifier used for clustering."""
    result = subprocess.run(
        [str(mmseqs_binary), "version"], capture_output=True, text=True, check=True
    )
    return (result.stdout or result.stderr).strip()


def load_cluster_assignment(cluster_tsv: str | Path) -> pd.DataFrame:
    """Parses mmseqs' <prefix>_cluster.tsv (representative_id, member_id per
    row, one row per cluster member including singletons) into a dense
    protein_id -> cluster_id table."""
    df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["representative_id", "protein_id"])
    # The representative accession is the stable cluster identifier. Integer
    # category codes depend on MMseqs2 row order and made seeded partitions
    # change across otherwise identical reruns.
    df["cluster_id"] = df["representative_id"].astype(str)
    return df[["protein_id", "cluster_id", "representative_id"]]
