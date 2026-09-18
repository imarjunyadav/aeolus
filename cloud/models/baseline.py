"""Transparent forecasting baselines used before Aeolus ML models exist."""
from __future__ import annotations

import numpy as np


def persistence_forecast(grid: np.ndarray, horizons: list[int]) -> list[np.ndarray]:
    """Carry the latest observed grid forward for each requested horizon."""
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError("grid must be 2-D")
    return [arr.copy() for _ in horizons]


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error for model-vs-baseline benchmarking."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError("actual and predicted arrays must have the same shape")
    return float(np.mean(np.abs(a - p)))
