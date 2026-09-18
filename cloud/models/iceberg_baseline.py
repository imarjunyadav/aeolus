"""Physics-light iceberg trajectory baseline for benchmarking.

Uses constant velocity from the latest two observations. This is a transparent
baseline, not a substitute for the eventual learned trajectory model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math


@dataclass(frozen=True)
class TrackPoint:
    timestamp: datetime
    latitude: float
    longitude: float


def _longitude_delta(a: float, b: float) -> float:
    return ((b - a + 180.0) % 360.0) - 180.0


def constant_velocity_forecast(history: list[TrackPoint], horizons_hours: list[int]) -> list[TrackPoint]:
    if len(history) < 2:
        raise ValueError("At least two iceberg observations are required")
    a, b = history[-2], history[-1]
    dt_h = (b.timestamp - a.timestamp).total_seconds() / 3600.0
    if dt_h <= 0:
        raise ValueError("Iceberg timestamps must increase")
    v_lat = (b.latitude - a.latitude) / dt_h
    v_lon = _longitude_delta(a.longitude, b.longitude) / dt_h
    out = []
    for h in horizons_hours:
        if h <= 0:
            raise ValueError("Forecast horizons must be positive")
        lat = max(-90.0, min(90.0, b.latitude + v_lat * h))
        lon = ((b.longitude + v_lon * h + 180.0) % 360.0) - 180.0
        out.append(TrackPoint(b.timestamp + timedelta(hours=h), lat, lon))
    return out
