import numpy as np
from datetime import datetime, timedelta, timezone

from cloud.models.evaluation import benchmark_sea_ice, chronological_split
from cloud.models.sea_ice_dataset import SeaIceSequence
from cloud.models.iceberg_baseline import TrackPoint, constant_velocity_forecast


def _sequence(t, offset):
    base = np.full((2, 2), offset, dtype=float)
    return SeaIceSequence(
        inputs=np.stack([base, base + 0.01, base + 0.02, base + 0.03]),
        target=base + 0.04,
        target_time=t,
    )


def test_chronological_split_keeps_newest_samples_in_test():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    seqs = [_sequence(start + timedelta(days=i), i / 100) for i in range(10)]
    train, test = chronological_split(seqs, 0.2)
    assert len(train) == 8
    assert len(test) == 2
    assert max(s.target_time for s in train) < min(s.target_time for s in test)


def test_benchmark_returns_finite_metrics():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    seqs = [_sequence(start + timedelta(days=i), i / 100) for i in range(10)]
    train, test = chronological_split(seqs)
    result = benchmark_sea_ice(train, test)
    assert result.sample_count == 2
    assert np.isfinite(result.persistence_mae)
    assert np.isfinite(result.trend_mae)


def test_iceberg_forecast_timestamps_are_future():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [
        TrackPoint(start, -70.0, 10.0),
        TrackPoint(start + timedelta(hours=1), -70.1, 10.2),
    ]
    result = constant_velocity_forecast(history, [6, 12])
    assert result[0].timestamp == start + timedelta(hours=7)
    assert result[1].timestamp == start + timedelta(hours=13)
