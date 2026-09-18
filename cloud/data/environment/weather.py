"""ECMWF IFS Open Data weather connector for Phase 3 Step 7.

The connector retrieves one IFS forecast step at 0.25 degrees and samples the
nearest grid point to the requested location. ECMWF Open Data separates
atmospheric and wave fields, so it makes two deliberately small requests:
``10u``, ``10v``, ``msl``, ``2t`` from the atmospheric stream and ``swh`` from
the wave stream. It does not assemble a ForecastPackage or implement GFS.

Timestamp semantics
-------------------
``source_data_timestamp`` is the IFS forecast initialisation time returned by
the Open Data client. ``valid_at`` is that initialisation plus the requested
forecast step. ``fetch_time_utc`` is retained in metadata only. In
particular, the time of the download is never used as the forecast valid time.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from cloud.data.exceptions import WeatherConnectorError
from shared.schemas.forecast import WeatherSnapshot


class EcmwfNetworkError(WeatherConnectorError):
    """ECMWF Open Data, its dependencies, or the network could not be reached."""


class EcmwfParseError(WeatherConnectorError):
    """The retrieved GRIB data cannot safely be represented as weather data."""


_ATMOSPHERIC_PARAMS = ["10u", "10v", "msl", "2t"]
_WAVE_PARAMS = ["swh"]
_MAX_FORECAST_HOURS = 72
_CYCLE_DELAY = timedelta(hours=9)
_VALID_STEPS_HOURS = 3


def _as_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC timestamp without silently shifting naïve UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _validate_location(lat: float, lon: float) -> tuple[float, float]:
    if not (math.isfinite(lat) and -90.0 <= lat <= 90.0):
        raise EcmwfParseError("latitude must be a finite value in [-90, 90]")
    if not math.isfinite(lon):
        raise EcmwfParseError("longitude must be finite")
    return float(lat), float(lon % 360.0)


def _latest_available_oper_cycle(now: datetime) -> datetime:
    """Choose the latest 00/12 UTC HRES cycle expected to be available."""
    eligible = _as_utc(now) - _CYCLE_DELAY
    hour = 12 if eligible.hour >= 12 else 0
    return eligible.replace(hour=hour, minute=0, second=0, microsecond=0)


def _request_for_valid_time(valid_at: datetime, now: datetime) -> tuple[datetime, int]:
    """Select an already-published 00/12 UTC cycle and an exact 3-hour step."""
    target = _as_utc(valid_at)
    source = _latest_available_oper_cycle(now)

    # A historical target needs a historical cycle. Open Data may no longer
    # retain it, but that should become a typed network error rather than a
    # silently incorrect source timestamp.
    if target < source:
        source_hour = 12 if target.hour >= 12 else 0
        source = target.replace(hour=source_hour, minute=0, second=0, microsecond=0)

    delta_seconds = (target - source).total_seconds()
    if delta_seconds < 0:
        raise EcmwfParseError("forecast valid time precedes the selected IFS cycle")
    if delta_seconds % (_VALID_STEPS_HOURS * 3600) != 0:
        raise EcmwfParseError("valid_at must fall on a 3-hour IFS forecast step")

    step = int(delta_seconds // 3600)
    if step > _MAX_FORECAST_HOURS:
        raise EcmwfParseError(
            f"valid_at is {step} hours after the selected cycle; Phase 3 supports 0-{_MAX_FORECAST_HOURS} hours"
        )
    return source, step


def _request(source: datetime | None, step: int, params: list[str], *, wave: bool) -> dict[str, Any]:
    request: dict[str, Any] = {"type": "fc", "step": step, "param": params}
    if source is not None:
        request.update({"date": source.strftime("%Y-%m-%d"), "time": source.hour})
    if wave:
        request["stream"] = "wave"
    return request


def _result_datetime(result: Any) -> datetime:
    value = getattr(result, "datetime", None)
    if not isinstance(value, datetime):
        raise EcmwfParseError("ECMWF client did not return a forecast initialisation time")
    return _as_utc(value)


def _open_grib(path: Path) -> list[Any]:
    try:
        import cfgrib
    except ImportError as exc:
        raise EcmwfNetworkError("cfgrib is not installed; install requirements-cloud.txt") from exc

    try:
        return list(cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}))
    except Exception as exc:
        raise EcmwfParseError(f"could not read ECMWF GRIB data: {exc}") from exc


def _find_data_array(datasets: list[Any], names: tuple[str, ...]) -> Any:
    for dataset in datasets:
        data_vars = getattr(dataset, "data_vars", {})
        for name in names:
            if name in data_vars:
                return dataset[name]
    raise EcmwfParseError(f"ECMWF GRIB is missing required variable {names[0]!r}")


def _normalise_source_longitude(longitudes: Any, requested: float) -> float:
    try:
        import numpy as np

        values = np.asarray(longitudes, dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("empty or non-finite longitude coordinate")

        # Aeolus requests use a [0, 360) longitude convention internally.
        # Convert signed requests before sampling a 0-360 ECMWF grid.
        # If the source grid instead uses [-180, 180], convert back to that
        # convention for sampling.
        requested_0360 = requested % 360.0
        if values.min() < 0.0 and requested_0360 > 180.0:
            return requested_0360 - 360.0
        return requested_0360
    except Exception as exc:
        raise EcmwfParseError(f"ECMWF longitude coordinate is invalid: {exc}") from exc


def _point_value(data_array: Any, lat: float, lon: float, label: str) -> tuple[float, float, float, str]:
    """Select one finite field value and report the actual sampled coordinates/unit."""
    try:
        import numpy as np

        if "latitude" not in data_array.coords or "longitude" not in data_array.coords:
            raise EcmwfParseError(f"ECMWF {label} field is missing latitude/longitude coordinates")
        source_lon = _normalise_source_longitude(data_array["longitude"].values, lon)
        selected = data_array.sel(latitude=lat, longitude=source_lon, method="nearest")
        values = np.asarray(selected.values, dtype=float).squeeze()
        if values.size != 1 or not np.isfinite(values.item()):
            raise EcmwfParseError(f"ECMWF {label} value at requested location is missing or invalid")
        unit = str(data_array.attrs.get("units", "")).strip()
        selected_lat = float(np.asarray(selected["latitude"].values).item())
        selected_lon = float(np.asarray(selected["longitude"].values).item())
        return float(values.item()), selected_lat, selected_lon, unit
    except EcmwfParseError:
        raise
    except Exception as exc:
        raise EcmwfParseError(f"could not sample ECMWF {label} at requested location: {exc}") from exc


def _require_unit(unit: str, accepted: set[str], label: str) -> str:
    canonical = unit.lower().replace(" ", "")
    if canonical not in accepted:
        raise EcmwfParseError(f"ECMWF {label} has unsupported units {unit!r}")
    return canonical


def _weather_snapshot_from_grib(
    atmospheric: list[Any],
    waves: list[Any],
    *,
    lat: float,
    lon: float,
    source_time: datetime,
    valid_at: datetime,
    fetch_time: datetime,
) -> WeatherSnapshot:
    """Convert two ECMWF GRIB groups into the existing point WeatherSnapshot schema."""
    u10, selected_lat, selected_lon, u_unit = _point_value(
        _find_data_array(atmospheric, ("u10", "10u")), lat, lon, "u10"
    )
    v10, _, _, v_unit = _point_value(
        _find_data_array(atmospheric, ("v10", "10v")), lat, lon, "v10"
    )
    msl, _, _, msl_unit = _point_value(
        _find_data_array(atmospheric, ("msl",)), lat, lon, "msl"
    )
    swh, _, _, swh_unit = _point_value(
        _find_data_array(waves, ("swh",)), lat, lon, "swh"
    )
    t2m, _, _, t2m_unit = _point_value(
        _find_data_array(atmospheric, ("t2m", "2t")), lat, lon, "t2m"
    )

    _require_unit(
        u_unit,
        {"m/s", "ms-1", "m.s-1", "m*s-1", "ms**-1"},
        "u10",
    )
    _require_unit(
        v_unit,
        {"m/s", "ms-1", "m.s-1", "m*s-1", "ms**-1"},
        "v10",
    )
    msl_unit = _require_unit(msl_unit, {"pa", "hpa"}, "msl")
    _require_unit(swh_unit, {"m"}, "swh")
    t2m_unit = _require_unit(t2m_unit, {"k", "c", "degc", "°c"}, "t2m")

    if not (80_000.0 <= msl <= 110_000.0 if msl_unit == "pa" else 800.0 <= msl <= 1100.0):
        raise EcmwfParseError("ECMWF msl is outside the plausible sea-level-pressure range")
    if not 0.0 <= swh <= 30.0:
        raise EcmwfParseError("ECMWF swh is outside the plausible wave-height range")
    if not (150.0 <= t2m <= 330.0 if t2m_unit == "k" else -123.15 <= t2m <= 56.85):
        raise EcmwfParseError("ECMWF t2m is outside the plausible air-temperature range")

    pressure_hpa = msl / 100.0 if msl_unit == "pa" else msl
    temperature_c = t2m - 273.15 if t2m_unit == "k" else t2m
    wind_speed = math.hypot(u10, v10)
    # Meteorological convention: direction the wind is coming *from*.
    wind_direction = (math.degrees(math.atan2(-u10, -v10)) + 360.0) % 360.0
    source_time = _as_utc(source_time)
    valid_at = _as_utc(valid_at)
    fetch_time = _as_utc(fetch_time)

    return WeatherSnapshot(
        source_data_timestamp=source_time,
        source_data_age_seconds=max(0, int((fetch_time - source_time).total_seconds())),
        valid_at=valid_at,
        wind_speed_ms=wind_speed,
        wind_direction_deg=wind_direction,
        pressure_hpa=pressure_hpa,
        significant_wave_height_m=swh,
        air_temperature_c=temperature_c,
        available_variables=["u10", "v10", "msl", "swh", "t2m"],
        metadata={
            "provider": "ECMWF IFS Open Data",
            "forecast_initialization_utc": source_time.isoformat(),
            "fetch_time_utc": fetch_time.isoformat(),
            "requested_latitude": lat,
            "requested_longitude": lon,
            "sampled_latitude": selected_lat,
            "sampled_longitude": selected_lon,
            "raw_units": {"u10": u_unit, "v10": v_unit, "msl": msl_unit, "swh": swh_unit, "t2m": t2m_unit},
        },
    )


def fetch_weather_forecast(
    lat: float,
    lon: float,
    valid_at: datetime | None = None,
) -> WeatherSnapshot:
    """Fetch and normalise one ECMWF IFS weather forecast sample.

    ``valid_at`` requests an exact 3-hour forecast validity time (up to 72 h
    from the selected available 00/12 UTC cycle). When omitted, the most
    recent ECMWF cycle's analysis step is fetched. Antarctica is supported
    because IFS Open Data is global; location bounds are still validated before
    the request and GRIB sampling.
    """
    lat, lon = _validate_location(lat, lon)
    fetch_time = datetime.now(tz=timezone.utc)

    if valid_at is None:
        requested_source: datetime | None = None
        step = 0
    else:
        requested_source, step = _request_for_valid_time(valid_at, fetch_time)

    try:
        from ecmwf.opendata import Client
    except ImportError as exc:
        raise EcmwfNetworkError("ecmwf-opendata is not installed; install requirements-cloud.txt") from exc

    try:
        with TemporaryDirectory(prefix="aeolus-ecmwf-") as temporary_dir:
            atmospheric_path = Path(temporary_dir) / "atmosphere.grib2"
            wave_path = Path(temporary_dir) / "wave.grib2"
            client = Client(source="ecmwf")
            atmospheric_result = client.retrieve(
                _request(requested_source, step, _ATMOSPHERIC_PARAMS, wave=False),
                str(atmospheric_path),
            )
            wave_result = client.retrieve(
                _request(requested_source, step, _WAVE_PARAMS, wave=True),
                str(wave_path),
            )
            source_time = _result_datetime(atmospheric_result)
            wave_source_time = _result_datetime(wave_result)
            if source_time != wave_source_time:
                raise EcmwfParseError("atmospheric and wave data came from different IFS cycles")
            actual_valid_at = source_time + timedelta(hours=step)
            if valid_at is not None and actual_valid_at != _as_utc(valid_at):
                raise EcmwfParseError("ECMWF returned a cycle/step that does not match requested valid_at")
            return _weather_snapshot_from_grib(
                _open_grib(atmospheric_path),
                _open_grib(wave_path),
                lat=lat,
                lon=lon,
                source_time=source_time,
                valid_at=actual_valid_at,
                fetch_time=datetime.now(tz=timezone.utc),
            )
    except WeatherConnectorError:
        raise
    except Exception as exc:
        raise EcmwfNetworkError(f"ECMWF Open Data request failed: {exc}") from exc
