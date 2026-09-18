"""Evaluation utilities for Aeolus forecasting baselines.

These functions use chronological hold-out evaluation so adjacent observations
are not randomly mixed between training and test sets.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .baseline import mae
from .sea_ice_forecaster import LinearTrendSeaIceForecaster
from .sea_ice_dataset import SeaIceSequence


@dataclass(frozen=True)
class ForecastBenchmark:
    horizon_steps: int
    sample_count: int
    persistence_mae: float
    trend_mae: float
    improvement_percent: float

    def as_dict(self) -> dict:
        return {
            "horizon_steps": self.horizon_steps,
            "sample_count": self.sample_count,
            "persistence_mae": self.persistence_mae,
            "linear_trend_mae": self.trend_mae,
            "improvement_percent": self.improvement_percent,
        }


def chronological_split(
    sequences: list[SeaIceSequence], test_fraction: float = 0.2
) -> tuple[list[SeaIceSequence], list[SeaIceSequence]]:
    """Split sequences chronologically; the newest samples become test data."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = sorted(sequences, key=lambda s: s.target_time)
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - test_fraction))))
    return ordered[:cut], ordered[cut:]


def benchmark_sea_ice(
    train: list[SeaIceSequence],
    test: list[SeaIceSequence],
    *,
    lead_steps: int = 1,
) -> ForecastBenchmark:
    """Fit the trend model on training windows and compare it with persistence."""
    if not train or not test:
        raise ValueError("train and test sets must both be non-empty")
    inputs_train = np.stack([s.inputs for s in train])
    inputs_test = np.stack([s.inputs for s in test])
    actual = np.stack([s.target for s in test])

    persistence = inputs_test[:, -1]
    persistence_error = mae(actual, persistence)

    model = LinearTrendSeaIceForecaster(inputs_train.shape[1]).fit(inputs_train)
    predicted = model.predict(inputs_test, lead_steps=lead_steps)
    trend_error = mae(actual, predicted)

    improvement = 0.0 if persistence_error == 0 else (
        (persistence_error - trend_error) / persistence_error * 100.0
    )
    return ForecastBenchmark(
        horizon_steps=lead_steps,
        sample_count=len(test),
        persistence_mae=persistence_error,
        trend_mae=trend_error,
        improvement_percent=improvement,
    )
