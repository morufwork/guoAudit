"""Experiment logging: console + a run.log file per script/experiment."""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | Path = "logs", filename: str | None = None) -> logging.Logger:
    """Configured logger that writes to both stdout and logs/<filename>.

    Idempotent: calling this again for the same `name` will not duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / (filename or f"{name}.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
