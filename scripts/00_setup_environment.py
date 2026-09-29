"""Environment and reproducibility report.

Writes results/environment_report.json and results/software_versions.csv.
Run once at project setup and again whenever the environment changes.
"""
import csv
import platform
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_config, save_json, dataset_hashes
from src.utils.logging import get_logger

logger = get_logger("setup_environment")

REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_gpu() -> dict:
    try:
        import torch

        available = torch.cuda.is_available()
        return {
            "torch_installed": True,
            "cuda_available": available,
            "device_count": torch.cuda.device_count() if available else 0,
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else [],
        }
    except ImportError:
        return {"torch_installed": False, "cuda_available": False, "device_count": 0, "device_names": []}


def main():
    config = load_config(REPO_ROOT / "configs" / "data.yaml")

    env_report = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu": detect_gpu(),
        "dataset_hashes": dataset_hashes(
            {
                "proteins_file": REPO_ROOT / config["proteins_file"],
                "pairs_file": REPO_ROOT / config["pairs_file"],
                "original_split_train": REPO_ROOT / config["original_split"]["train_file"],
                "original_split_test": REPO_ROOT / config["original_split"]["test_file"],
            }
        ),
    }
    save_json(env_report, REPO_ROOT / "results" / "environment_report.json")
    logger.info("Wrote results/environment_report.json")

    versions_path = REPO_ROOT / "results" / "software_versions.csv"
    versions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(versions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["package", "version"])
        for dist in sorted(distributions(), key=lambda d: d.metadata["Name"] or ""):
            name = dist.metadata["Name"]
            if name:
                writer.writerow([name, dist.version])
    logger.info("Wrote results/software_versions.csv")

    logger.info(f"GPU: {env_report['gpu']}")
    logger.info(f"Dataset hashes: {env_report['dataset_hashes']}")


if __name__ == "__main__":
    main()
