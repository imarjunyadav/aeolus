"""Small, dependency-light trainable sea-ice forecasting baseline.

This model is intentionally simple: for each grid cell it fits a least-squares
trend over the recent input frames and extrapolates one or more future steps.
It gives Aeolus a genuine learned/statistical baseline without committing the
architecture to a deep-learning framework before dataset validation.
"""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np


class LinearTrendSeaIceForecaster:
    """Per-cell linear trend forecaster over recent concentration frames."""

    def __init__(self, input_steps: int):
        if input_steps < 2:
            raise ValueError("input_steps must be at least 2")
        self.input_steps = int(input_steps)
        self.slopes: np.ndarray | None = None
        self.intercepts: np.ndarray | None = None

    def fit(self, inputs: np.ndarray) -> "LinearTrendSeaIceForecaster":
        """Fit trends to ``inputs`` shaped [samples, steps, lat, lon]."""
        x = np.asarray(inputs, dtype=float)
        if x.ndim != 4 or x.shape[1] != self.input_steps:
            raise ValueError("inputs must be [samples, input_steps, lat, lon]")
        # Collapse samples into a single collection of local temporal windows.
        t = np.arange(self.input_steps, dtype=float)
        tc = t - t.mean()
        denom = float(np.sum(tc * tc))
        centered = x - np.mean(x, axis=1, keepdims=True)
        sample_slopes = np.sum(centered * tc.reshape(1, -1, 1, 1), axis=1) / denom
        self.slopes = np.mean(sample_slopes, axis=0)
        self.intercepts = np.mean(x[:, -1], axis=0) - self.slopes * (self.input_steps - 1)
        return self

    def predict(self, inputs: np.ndarray, lead_steps: int = 1) -> np.ndarray:
        """Predict future concentration, returning [samples, lat, lon]."""
        if self.slopes is None or self.intercepts is None:
            raise RuntimeError("Model must be fit before prediction")
        x = np.asarray(inputs, dtype=float)
        if x.ndim != 4 or x.shape[1] != self.input_steps:
            raise ValueError("inputs must be [samples, input_steps, lat, lon]")
        if lead_steps < 1:
            raise ValueError("lead_steps must be positive")
        t_future = float(self.input_steps - 1 + lead_steps)
        last = x[:, -1]
        # Blend the global fitted trend with each sample's last observed level.
        delta = self.slopes * float(lead_steps)
        return np.clip(last + delta, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"input_steps": self.input_steps, "slopes": self.slopes}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "LinearTrendSeaIceForecaster":
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        obj = cls(int(state["input_steps"]))
        obj.slopes = np.asarray(state["slopes"], dtype=float)
        obj.intercepts = np.zeros_like(obj.slopes)
        return obj
