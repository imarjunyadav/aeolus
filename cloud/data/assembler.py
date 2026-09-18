"""ForecastPackage assembler for MOCK and REAL modes.

REAL mode is now a Phase-3 baseline assembler.  It does not pretend that
persistence is an ML forecast: current sea-ice concentration and iceberg
positions are carried forward as a deterministic baseline with growing
uncertainty.  Weather and ocean remain provider forecasts/analyses.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

from shared.schemas.common import GridMetadata, Position
from shared.schemas.forecast import (
    ForecastPackage,
    IcebergForecast,
    IcebergTrajectoryHorizon,
    SeaIceForecast,
    SeaIceForecastHorizon,
)
from shared.schemas.vessel import ForecastRequest

from .modes import DataMode, get_data_mode


def _area(request: ForecastRequest) -> GridMetadata:
    if request.route_area:
        a = request.route_area
        if all(k in a for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
            return GridMetadata(
                lat_min=float(a["lat_min"]), lat_max=float(a["lat_max"]),
                lon_min=float(a["lon_min"]), lon_max=float(a["lon_max"]),
                n_lat=int(a.get("n_lat", 20)), n_lon=int(a.get("n_lon", 20)),
            )
    p = request.vessel_position
    return GridMetadata(
        lat_min=max(-90.0, p.latitude - 10.0),
        lat_max=min(-55.0, p.latitude + 10.0),
        lon_min=p.longitude - 10.0,
        lon_max=p.longitude + 10.0,
        n_lat=20,
        n_lon=20,
    )


def _persistence_iceberg(iceberg: IcebergForecast, horizons: list[int], now: datetime) -> IcebergForecast:
    trajectory = []
    for h in horizons:
        trajectory.append(IcebergTrajectoryHorizon(
            horizon_hours=h,
            valid_at=now + timedelta(hours=h),
            latitude=iceberg.last_known_position.latitude,
            longitude=iceberg.last_known_position.longitude,
            uncertainty_radius_km=15.0 + 2.5 * np.sqrt(h),
        ))
    return iceberg.model_copy(update={"horizons": trajectory})


def _real_package(request: ForecastRequest) -> ForecastPackage:
    from cloud.data.environment.weather import fetch_weather_forecast
    from cloud.data.icebergs.usnic import fetch_current_positions
    from cloud.data.environment.ocean import fetch_ocean_snapshot
    from cloud.data.sea_ice.nsidc import fetch_concentration_geotiff
    from cloud.data.regrid import regrid_to_package_grid

    now = datetime.now(tz=timezone.utc)
    horizons = sorted(set(request.forecast_horizons_hours or [6, 12, 24, 48, 72]))
    if any(h <= 0 or h > 72 for h in horizons):
        raise ValueError("REAL mode currently supports forecast horizons from 1 to 72 hours")

    grid = _area(request)
    target_date = (now - timedelta(days=1)).date()
    concentration_source = fetch_concentration_geotiff(target_date)
    concentration = np.asarray(regrid_to_package_grid(
        concentration_source, grid, "", source_crs="EPSG:3031", x_dim="x", y_dim="y", method="nearest"
    ), dtype=float)
    concentration = np.nan_to_num(concentration, nan=1.0, posinf=1.0, neginf=0.0)
    concentration = np.clip(concentration, 0.0, 1.0)

    age = max(0, int((now - datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)).total_seconds()))
    ice_horizons = []
    for h in horizons:
        # Explicit Phase-3 baseline, not ML. Uncertainty increases with lead time and data age.
        base_unc = min(0.5, 0.05 + age / 86400.0 * 0.03 + np.sqrt(h) * 0.015)
        uncertainty = np.full_like(concentration, base_unc, dtype=float)
        ice_horizons.append(SeaIceForecastHorizon(
            horizon_hours=h,
            valid_at=now + timedelta(hours=h),
            concentration_grid=concentration.tolist(),
            uncertainty_grid=uncertainty.tolist(),
        ))

    sea_ice = SeaIceForecast(
        source_data_timestamp=datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc),
        source_data_age_seconds=age,
        grid_metadata=grid,
        horizons=ice_horizons,
        quality_flags={
            "provider": "NSIDC G02135",
            "forecast_method": "persistence_baseline",
            "ml_forecast": "false",
        },
    )

    icebergs, iceberg_timestamp = fetch_current_positions()
    icebergs = [_persistence_iceberg(i, horizons, now) for i in icebergs]

    weather = fetch_weather_forecast(request.vessel_position.latitude, request.vessel_position.longitude, now)
    npdc_url = os.getenv("AEOLUS_NPDC_EXPORT_URL")
    npdc_path = os.getenv("AEOLUS_NPDC_EXPORT_PATH")
    if npdc_url or npdc_path:
        from cloud.data.india.npdc import fetch_npdc_export, load_npdc_file
        indian = fetch_npdc_export(npdc_url) if npdc_url else load_npdc_file(npdc_path)
        if indian:
            # NPDC station observations are retained as provenance/validation data;
            # they are not mislabeled as a point forecast for the vessel.
            closest = min(indian, key=lambda x: abs((x.valid_at - weather.valid_at).total_seconds()))
            weather = weather.model_copy(update={
                "metadata": {
                    **weather.metadata,
                    "indian_npdc_observation": closest.model_dump(mode="json"),
                    "indian_data_role": "independent_station_observation",
                }
            })

    ocean = fetch_ocean_snapshot(grid, valid_at=datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc))

    # Confidence is intentionally conservative while ML and forecast-error calibration are pending.
    confidence = max(0.1, min(0.85, 0.85 * np.exp(-age / (3 * 86400))))
    return ForecastPackage(
        package_id=f"FP-{uuid.uuid4().hex}",
        schema_version="1.1",
        generated_at=now,
        generated_by="aeolus-real-assembler/1.1",
        valid_from=now,
        valid_to=now + timedelta(hours=max(horizons)),
        request_area={"lat_min": grid.lat_min, "lat_max": grid.lat_max, "lon_min": grid.lon_min, "lon_max": grid.lon_max},
        sea_ice=sea_ice,
        icebergs=icebergs,
        weather=weather,
        ocean=ocean,
        overall_confidence=float(confidence),
    )


def build_forecast_package(request: ForecastRequest, mode: DataMode | None = None) -> ForecastPackage:
    if mode is None:
        mode = get_data_mode()
    if mode == DataMode.MOCK:
        from data.mock.generator import generate_mock_forecast_package
        return generate_mock_forecast_package(
            vessel_position=request.vessel_position,
            destination=request.destination,
            forecast_horizons_hours=request.forecast_horizons_hours,
        )
    return _real_package(request)
