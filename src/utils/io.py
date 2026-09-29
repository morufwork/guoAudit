"""Shared I/O helpers: config loading, JSON/CSV writing, file hashing."""
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_hashes(paths: dict[str, str]) -> dict[str, str]:
    """Map {label: filepath} -> {label: sha256} for files that exist."""
    hashes = {}
    for label, p in paths.items():
        if Path(p).exists():
            hashes[label] = sha256_file(p)
    return hashes
