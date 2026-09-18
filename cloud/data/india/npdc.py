"""Adapter for exported/downloaded NPDC Antarctic station observations.

NPDC is an authoritative Indian polar-data portal.  Some observations are
available through request/search workflows rather than a stable public API.
Aeolus therefore accepts an NPDC export as CSV/JSON or a configured HTTPS URL
instead of inventing an undocumented API endpoint.

The adapter normalizes the common meteorological fields into the existing
WeatherSnapshot schema.  It is intentionally transport-agnostic so a future
NPDC download client can feed the same normalizer without changing the rest of
Aeolus.
"""

from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.request import Request, urlopen

from cloud.data.exceptions import WeatherConnectorError
from shared.schemas.forecast import WeatherSnapshot


class NpdcParseError(WeatherConnectorError):
    """NPDC export is malformed or contains no usable observation."""


class NpdcNetworkError(WeatherConnectorError):
    """Configured NPDC download could not be reached."""


FIELD_ALIASES = {
    "timestamp": ("timestamp", "datetime", "date_time", "date", "time", "observation_time"),
    "station": ("station", "station_name", "site", "location"),
    "temperature": ("temperature", "temp", "air_temperature", "air_temp", "temperature_c"),
    "humidity": ("relative_humidity", "rh", "humidity", "relative_humidity_pct"),
    "pressure": ("pressure", "mslp", "mslp_hpa", "pressure_hpa", "surface_pressure"),
    "wind_speed": ("wind_speed", "wind_speed_ms", "ws", "windspeed"),
    "wind_direction": ("wind_direction", "wind_direction_deg", "wd", "winddir"),
}


def _key_map(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {str(k).strip().lower().replace(" ", "_"): v for k, v in row.items()}
    return normalized


def _get(row: dict[str, Any], field: str) -> Any:
    for alias in FIELD_ALIASES[field]:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _float(value: Any, field: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NpdcParseError(f"NPDC {field} value {value!r} is not numeric") from exc
    if not math.isfinite(result):
        raise NpdcParseError(f"NPDC {field} value is not finite")
    return result


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            # Common Indian station export formats.
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise NpdcParseError(f"NPDC timestamp {value!r} is not parseable")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalise_row(raw: dict[str, Any], received_at: datetime) -> WeatherSnapshot:
    row = _key_map(raw)
    timestamp_value = _get(row, "timestamp")
    if timestamp_value is None:
        raise NpdcParseError("NPDC observation is missing a timestamp")
    source_time = _parse_timestamp(timestamp_value)
    age = max(0, int((received_at - source_time).total_seconds()))

    temp = _float(_get(row, "temperature"), "temperature")
    pressure = _float(_get(row, "pressure"), "pressure")
    wind_speed = _float(_get(row, "wind_speed"), "wind_speed")
    wind_direction = _float(_get(row, "wind_direction"), "wind_direction")
    humidity = _float(_get(row, "humidity"), "relative_humidity")

    if wind_direction is not None and not 0 <= wind_direction <= 360:
        raise NpdcParseError("NPDC wind direction must be in [0, 360]")
    if humidity is not None and not 0 <= humidity <= 100:
        raise NpdcParseError("NPDC relative humidity must be in [0, 100]")

    available: list[str] = []
    for name, value in (
        ("air_temperature_c", temp),
        ("pressure_hpa", pressure),
        ("wind_speed_ms", wind_speed),
        ("wind_direction_deg", wind_direction),
    ):
        if value is not None:
            available.append(name)

    metadata = {
        "provider": "NCPOR / NPDC",
        "source_id": "india.npdc.antarctic_weather",
        "station": _get(row, "station"),
        "relative_humidity_pct": humidity,
        "access": "NPDC export/download",
    }
    return WeatherSnapshot(
        source_data_timestamp=source_time,
        source_data_age_seconds=age,
        valid_at=source_time,
        wind_speed_ms=wind_speed,
        wind_direction_deg=wind_direction,
        pressure_hpa=pressure,
        air_temperature_c=temp,
        available_variables=available,
        metadata=metadata,
    )


def parse_npdc_rows(rows: Iterable[dict[str, Any]], received_at: datetime | None = None) -> list[WeatherSnapshot]:
    received = received_at or datetime.now(tz=timezone.utc)
    snapshots = []
    for row in rows:
        try:
            snapshots.append(_normalise_row(row, received))
        except NpdcParseError:
            continue
    if not snapshots:
        raise NpdcParseError("NPDC export contained no usable meteorological observations")
    return snapshots


def parse_npdc_csv(text: str, received_at: datetime | None = None) -> list[WeatherSnapshot]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise NpdcParseError("NPDC CSV has no header")
    return parse_npdc_rows(reader, received_at)


def parse_npdc_json(payload: str | bytes | list[dict[str, Any]] | dict[str, Any], received_at: datetime | None = None) -> list[WeatherSnapshot]:
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise NpdcParseError(f"NPDC JSON is invalid: {exc}") from exc
    if isinstance(payload, dict):
        rows = payload.get("data", payload.get("observations", payload.get("results")))
        if rows is None:
            rows = [payload]
    else:
        rows = payload
    if not isinstance(rows, list):
        raise NpdcParseError("NPDC JSON does not contain an observation list")
    return parse_npdc_rows(rows, received_at)


def load_npdc_file(path: str, received_at: datetime | None = None) -> list[WeatherSnapshot]:
    """Load an NPDC CSV/JSON export already downloaded from the official portal."""
    try:
        body = open(path, "rb").read()
    except OSError as exc:
        raise NpdcNetworkError(f"NPDC local file could not be read: {exc}") from exc
    if path.lower().endswith(".json"):
        return parse_npdc_json(body, received_at)
    return parse_npdc_csv(body.decode("utf-8-sig"), received_at)


def fetch_npdc_export(url: str, *, timeout: int = 30) -> list[WeatherSnapshot]:
    """Fetch a configured NPDC CSV/JSON export URL and normalize it."""
    try:
        request = Request(url, headers={"User-Agent": "aeolus-npdc-adapter/1.0"})
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "").lower()
    except Exception as exc:
        raise NpdcNetworkError(f"NPDC export download failed: {exc}") from exc

    if "json" in content_type or url.lower().endswith(".json"):
        return parse_npdc_json(body)
    return parse_npdc_csv(body.decode("utf-8-sig"))
