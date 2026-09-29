"""Load the Guo yeast benchmark files into pandas DataFrames.

Source files are headerless TSVs (see configs/data.yaml for the schema).
"""
from pathlib import Path

import pandas as pd


def load_proteins(config: dict, repo_root: str | Path = ".") -> pd.DataFrame:
    path = Path(repo_root) / config["proteins_file"]
    return pd.read_csv(path, sep="\t", header=None, names=config["proteins_columns"])


def load_pairs(config: dict, repo_root: str | Path = ".") -> pd.DataFrame:
    path = Path(repo_root) / config["pairs_file"]
    return pd.read_csv(path, sep="\t", header=None, names=config["pairs_columns"])


def load_original_split(config: dict, repo_root: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The benchmark's pre-supplied train/test split.

    WARNING: this split has protein overlap between train and test — it is
    the naive random split,
    not a leakage-resistant split. Do not use it for R1-R3 or homology
    generalization claims.
    """
    cols = config["pairs_columns"]
    train_path = Path(repo_root) / config["original_split"]["train_file"]
    test_path = Path(repo_root) / config["original_split"]["test_file"]
    train = pd.read_csv(train_path, sep="\t", header=None, names=cols)
    test = pd.read_csv(test_path, sep="\t", header=None, names=cols)
    return train, test
