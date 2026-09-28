"""Utility functions for seeding and metrics."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across python random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Compute moving average of a 1D array."""
    if len(values) == 0:
        return np.array([])
    if len(values) < window:
        window = max(1, len(values))
    # ponytail: simple cumulative sum moving average
    ret = np.cumsum(values, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window
