"""
Mock ForecastPackage generator.

Purpose: produce structurally valid ForecastPackage objects so that
the vessel navigation engine (Track B) can be built and tested without
waiting for real data or trained ML models.

The generated data is synthetic and not geophysically realistic.
It is intentionally minimal. Replace with real ML inference once
Track A (cloud + ML) is ready.
"""

from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timedelta, timezone

from shared.schemas.common import GridMetadata, Position
from shared.schemas.forecast import (
    ForecastPackage,
    IcebergForecast,
    IcebergTrajectoryHorizon,
    OceanSnapshot,
    SeaIceForecast,
    SeaIceForecastHorizon,
    WeatherSnapshot,
)

# Default forecast horizons in hours
DEFAULT_HORIZONS = [6, 12, 24, 48, 72]

# Grid size: 20×20 cells at ~1° resolution covers a 20°×20° area.
# Coarse but sufficient to exercise the navigation algorithms.
GRID_SIZE = 20

# Approximate iceberg drift in the Southern Ocean: ~0.2–0.5 deg/day eastward.
# Values are placeholder; actual drift depends on current + wind.
ICEBERG_DRIFT_LON_DEG_PER_HOUR = 0.015
ICEBERG_DRIFT_LAT_DEG_PER_HOUR = 0.003


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_sea_ice_grid(
    n_lat: int,
    n_lon: int,
    lat_min: float,
    lat_max: float,
    rng: random.Random,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Generate a synthetic sea-ice concentration and uncertainty grid.

    Concentration increases toward lower latitudes (higher abs values)
    in a simplified Antarctic model, with added noise.
    """
    concentration: list[list[float]] = []
    uncertainty: list[list[float]] = []

    for i in range(n_lat):
        lat = lat_min + (lat_max - lat_min) * i / max(n_lat - 1, 1)
        conc_row: list[float] = []
        unc_row: list[float] = []
        for _ in range(n_lon):
            # Simplified: more ice at more southerly (more negative) latitudes.
            base = max(0.0, min(1.0, (-lat - 60.0) / 20.0))
            noise = rng.gauss(0.0, 0.08)
            conc = max(0.0, min(1.0, base + noise))
            unc = max(0.02, min(0.3, rng.gauss(0.08, 0.03)))
            conc_row.append(round(conc, 4))
            unc_row.append(round(unc, 4))
        concentration.append(conc_row)
        uncertainty.append(unc_row)

    return concentration, uncertainty


def _make_iceberg_forecast(
    iceberg_id: str,
    position: Position,
    now: datetime,
    horizons: list[int],
    rng: random.Random,
) -> IcebergForecast:
    """
    Generate a synthetic iceberg trajectory using simple linear drift.
    Uncertainty radius grows with horizon (uncertainty compounds over time).
    """
    trajectory: list[IcebergTrajectoryHorizon] = []
    for h in horizons:
        future_lon = position.longitude + ICEBERG_DRIFT_LON_DEG_PER_HOUR * h
        future_lat = position.latitude + ICEBERG_DRIFT_LAT_DEG_PER_HOUR * h + rng.gauss(0, 0.05)
        # Uncertainty grows roughly with sqrt(horizon) — placeholder scaling.
        uncertainty_km = round(5.0 + 2.5 * math.sqrt(h), 1)
        trajectory.append(
            IcebergTrajectoryHorizon(
                horizon_hours=h,
                valid_at=now + timedelta(hours=h),
                latitude=round(future_lat, 6),
                longitude=round(future_lon % 360, 6),
                uncertainty_radius_km=uncertainty_km,
            )
        )
    return IcebergForecast(
        iceberg_id=iceberg_id,
        last_known_position=position,
        last_known_timestamp=now,
        source_data_age_seconds=0,
        horizons=trajectory,
    )


def generate_mock_forecast_package(
    vessel_position: Position,
    destination: Position | None = None,
    forecast_horizons_hours: list[int] | None = None,
    n_icebergs: int = 4,
    seed: int | None = None,
    reference_time: datetime | None = None,
) -> ForecastPackage:
    """
    Generate a minimal but structurally complete ForecastPackage.

    Parameters
    ----------
    vessel_position : Position
        Center of the forecast area.
    destination : Position, optional
        Not used in mock generation; accepted to match ForecastRequest signature.
        ForecastRequest.route_area is similarly accepted by the schema but not
        consumed here — Phase 1 generates a fixed ±10° area around the vessel.
    forecast_horizons_hours : list[int], optional
        Defaults to [6, 12, 24, 48, 72].
    n_icebergs : int
        Number of synthetic icebergs to generate (default 4).
    seed : int, optional
        Controls all random values (grid, icebergs, weather, ocean).
        package_id always uses uuid4() regardless of seed — each package
        needs a unique ID at runtime.
    reference_time : datetime, optional
        Base timestamp for all valid_at and horizon fields.  When provided,
        the package is fully deterministic given a fixed seed (useful for
        testing and benchmarking).  Defaults to the current UTC time.
    """
    horizons = forecast_horizons_hours or DEFAULT_HORIZONS
    rng = random.Random(seed)
    now = reference_time if reference_time is not None else _now_utc()

    # --- Grid ---
    # Centre the grid on vessel position; extend ±10° lat, ±10° lon.
    lat_center = vessel_position.latitude
    lon_center = vessel_position.longitude
    lat_min = max(-90.0, lat_center - 10.0)
    lat_max = min(-55.0, lat_center + 10.0)  # cap north at -55° (Antarctic focus)
    lon_min = lon_center - 10.0
    lon_max = lon_center + 10.0

    grid_meta = GridMetadata(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        n_lat=GRID_SIZE,
        n_lon=GRID_SIZE,
    )

    ice_horizons: list[SeaIceForecastHorizon] = []
    for h in horizons:
        conc, unc = _make_sea_ice_grid(GRID_SIZE, GRID_SIZE, lat_min, lat_max, rng)
        ice_horizons.append(
            SeaIceForecastHorizon(
                horizon_hours=h,
                valid_at=now + timedelta(hours=h),
                concentration_grid=conc,
                uncertainty_grid=unc,
            )
        )

    sea_ice = SeaIceForecast(
        source_data_timestamp=now,
        source_data_age_seconds=0,
        grid_metadata=grid_meta,
        horizons=ice_horizons,
    )

    # --- Icebergs ---
    icebergs: list[IcebergForecast] = []
    for i in range(n_icebergs):
        pos = Position(
            latitude=round(rng.uniform(lat_min, lat_max), 4),
            longitude=round(rng.uniform(lon_min, lon_max), 4),
        )
        icebergs.append(
            _make_iceberg_forecast(
                iceberg_id=f"MOCK-{i+1:04d}",
                position=pos,
                now=now,
                horizons=horizons,
                rng=rng,
            )
        )

    # --- Weather snapshot ---
    # Rough Southern Ocean values; not geophysically tuned.
    weather = WeatherSnapshot(
        source_data_timestamp=now,
        source_data_age_seconds=0,
        valid_at=now,
        wind_speed_ms=round(rng.uniform(8.0, 25.0), 1),
        wind_direction_deg=round(rng.uniform(0.0, 360.0), 1),
        pressure_hpa=round(rng.uniform(970.0, 1005.0), 1),
        significant_wave_height_m=round(rng.uniform(2.0, 6.0), 2),
        air_temperature_c=round(rng.uniform(-25.0, -5.0), 1),
        available_variables=[
            "wind_speed_ms",
            "wind_direction_deg",
            "pressure_hpa",
            "significant_wave_height_m",
            "air_temperature_c",
        ],
    )

    # --- Ocean snapshot ---
    # Antarctic Circumpolar Current is primarily eastward; ~0.1–0.4 m/s.
    ocean = OceanSnapshot(
        source_data_timestamp=now,
        source_data_age_seconds=0,
        valid_at=now,
        current_u_ms=round(rng.uniform(0.05, 0.4), 3),
        current_v_ms=round(rng.gauss(0.0, 0.1), 3),
        sst_c=round(rng.uniform(-2.0, 4.0), 2),
        available_variables=["current_u_ms", "current_v_ms", "sst_c"],
    )

    valid_to = now + timedelta(hours=max(horizons))

    return ForecastPackage(
        package_id=str(uuid.uuid4()),
        generated_at=now,
        generated_by="aeolus-mock-generator/1.0",
        valid_from=now,
        valid_to=valid_to,
        request_area={
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        sea_ice=sea_ice,
        icebergs=icebergs,
        weather=weather,
        ocean=ocean,
        overall_confidence=0.95,
    )
