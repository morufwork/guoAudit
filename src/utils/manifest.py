"""Experiment manifest: records what produced a given results/ directory.

Every experiment directory should get one of these so results are regenerable without retraining.
"""
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any

from src.utils.io import save_json


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def _package_versions(packages: list[str]) -> dict[str, str]:
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def build_manifest(config: dict[str, Any], seed: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "config": config,
        "package_versions": _package_versions(
            ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "networkx", "biopython"]
        ),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(run_dir: str | Path, config: dict[str, Any], seed: int | None = None, extra: dict[str, Any] | None = None) -> None:
    manifest = build_manifest(config, seed=seed, extra=extra)
    save_json(manifest, Path(run_dir) / "environment.json")
