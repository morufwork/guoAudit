"""Global random seed handling for reproducibility."""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed all RNGs this project touches. Call once at the start of every script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
