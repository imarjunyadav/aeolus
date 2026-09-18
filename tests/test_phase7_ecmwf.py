"""Focused tests for the Phase 3 ECMWF IFS Open Data connector.

Fixture tests use in-memory xarray data. The sole live test requests one
Antarctic point at forecast step 0 and is explicitly skipped when ECMWF cannot
be contacted; a skip is an UNVERIFIED integration result, not a mock success.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from cloud.data.exceptions import WeatherConnectorError
from cloud.data.environment.weather import (
    EcmwfNetworkError,
    EcmwfParseError,
    _request_for_valid_time,
    _validate_location,
    _weather_snapshot_from_grib,
    fetch_weather_forecast,
)
from shared.schemas.forecast import WeatherSnapshot


_SOURCE_TIME = datetime(2026, 9, 15, 0, tzinfo=timezone.utc)
_VALID_TIME = _SOURCE_TIME + timedelta(hours=6)
_FETCH_TIME = _VALID_TIME + timedelta(minutes=10)


def _datasets(*, longitude: float = 190.0, msl_unit: str = "Pa", t2m: float = 273.15) -> tuple[list[xr.Dataset], list[xr.Dataset]]:
    coords = {"latitude": [-76.0, -75.0], "longitude": [180.0, longitude]}
    atmospheric = xr.Dataset(
        {
            "u10": (("latitude", "longitude"), np.array([[3.0, 3.0], [3.0, 3.0]]), {"units": "m s-1"}),
            "v10": (("latitude", "longitude"), np.array([[4.0, 4.0], [4.0, 4.0]]), {"units": "m s-1"}),
            "msl": (("latitude", "longitude"), np.array([[101325.0, 101325.0], [101325.0, 101325.0]]), {"units": msl_unit}),
            "t2m": (("latitude", "longitude"), np.array([[t2m, t2m], [t2m, t2m]]), {"units": "K"}),
        },
        coords=coords,
    )
    waves = xr.Dataset(
        {"swh": (("latitude", "longitude"), np.array([[2.5, 2.5], [2.5, 2.5]]), {"units": "m"})},
        coords=coords,
    )
    return [atmospheric], [waves]


class TestSnapshotParsing:
    def test_normalises_all_five_variables_and_timestamps(self):
        atmospheric, waves = _datasets()
        snapshot = _weather_snapshot_from_grib(
            atmospheric, waves, lat=-75.1, lon=-170.2,
            source_time=_SOURCE_TIME, valid_at=_VALID_TIME, fetch_time=_FETCH_TIME,
        )

        assert snapshot.wind_speed_ms == pytest.approx(5.0)
        assert snapshot.wind_direction_deg == pytest.approx(216.8698976)
        assert snapshot.pressure_hpa == pytest.approx(1013.25)
        assert snapshot.significant_wave_height_m == pytest.approx(2.5)
        assert snapshot.air_temperature_c == pytest.approx(0.0)
        assert snapshot.available_variables == ["u10", "v10", "msl", "swh", "t2m"]
        assert snapshot.source_data_timestamp == _SOURCE_TIME
        assert snapshot.valid_at == _VALID_TIME
        assert snapshot.source_data_timestamp != snapshot.valid_at
        assert snapshot.source_data_age_seconds == int((_FETCH_TIME - _SOURCE_TIME).total_seconds())
        assert snapshot.metadata["fetch_time_utc"] == _FETCH_TIME.isoformat()
        assert snapshot.metadata["sampled_longitude"] == pytest.approx(190.0)

    def test_supports_signed_source_longitudes_for_antarctica(self):
        atmospheric, waves = _datasets(longitude=-170.0)
        snapshot = _weather_snapshot_from_grib(
            atmospheric, waves, lat=-75.1, lon=190.0,
            source_time=_SOURCE_TIME, valid_at=_VALID_TIME, fetch_time=_FETCH_TIME,
        )
        assert snapshot.metadata["sampled_longitude"] == pytest.approx(-170.0)

    def test_missing_required_variable_is_parse_error(self):
        atmospheric, waves = _datasets()
        atmospheric[0] = atmospheric[0].drop_vars("t2m")
        with pytest.raises(EcmwfParseError, match="t2m"):
            _weather_snapshot_from_grib(
                atmospheric, waves, lat=-75.0, lon=180.0,
                source_time=_SOURCE_TIME, valid_at=_VALID_TIME, fetch_time=_FETCH_TIME,
            )

    def test_non_finite_value_is_parse_error(self):
        atmospheric, waves = _datasets()
        atmospheric[0]["u10"][:] = np.nan
        with pytest.raises(EcmwfParseError, match="u10.*missing or invalid"):
            _weather_snapshot_from_grib(
                atmospheric, waves, lat=-75.0, lon=180.0,
                source_time=_SOURCE_TIME, valid_at=_VALID_TIME, fetch_time=_FETCH_TIME,
            )

    def test_unsupported_unit_is_parse_error(self):
        atmospheric, waves = _datasets(msl_unit="kPa")
        with pytest.raises(EcmwfParseError, match="unsupported units"):
            _weather_snapshot_from_grib(
                atmospheric, waves, lat=-75.0, lon=180.0,
                source_time=_SOURCE_TIME, valid_at=_VALID_TIME, fetch_time=_FETCH_TIME,
            )


class TestRequestValidation:
    def test_normalises_longitude_and_accepts_antarctic_latitude(self):
        assert _validate_location(-89.5, -180.0) == (-89.5, 180.0)

    @pytest.mark.parametrize("lat, lon", [(float("nan"), 0.0), (-90.1, 0.0), (0.0, float("inf"))])
    def test_rejects_invalid_coordinates(self, lat: float, lon: float):
        with pytest.raises(EcmwfParseError):
            _validate_location(lat, lon)

    def test_selects_exact_valid_step(self):
        now = datetime(2026, 9, 16, 15, tzinfo=timezone.utc)
        source, step = _request_for_valid_time(datetime(2026, 9, 16, 18, tzinfo=timezone.utc), now)
        assert source == datetime(2026, 9, 16, 0, tzinfo=timezone.utc)
        assert step == 18

    def test_rejects_non_ifs_step(self):
        now = datetime(2026, 9, 16, 15, tzinfo=timezone.utc)
        with pytest.raises(EcmwfParseError, match="3-hour"):
            _request_for_valid_time(datetime(2026, 9, 16, 17, tzinfo=timezone.utc), now)


class TestFetchUnit:
    def test_requests_atmosphere_and_wave_stream_and_returns_snapshot(self):
        atmospheric, waves = _datasets()
        result = types.SimpleNamespace(datetime=_SOURCE_TIME)
        client = MagicMock()
        client.retrieve.return_value = result
        opendata = types.ModuleType("ecmwf.opendata")
        opendata.Client = MagicMock(return_value=client)
        ecmwf = types.ModuleType("ecmwf")
        ecmwf.opendata = opendata

        with patch.dict(sys.modules, {"ecmwf": ecmwf, "ecmwf.opendata": opendata}), \
             patch("cloud.data.environment.weather._open_grib", side_effect=[atmospheric, waves]):
            snapshot = fetch_weather_forecast(-75.0, 180.0)

        assert snapshot.source_data_timestamp == _SOURCE_TIME
        assert snapshot.valid_at == _SOURCE_TIME
        assert client.retrieve.call_count == 2
        atmospheric_request = client.retrieve.call_args_list[0].args[0]
        wave_request = client.retrieve.call_args_list[1].args[0]
        assert atmospheric_request["param"] == ["10u", "10v", "msl", "2t"]
        assert wave_request == {"type": "fc", "step": 0, "param": ["swh"], "stream": "wave"}

    def test_client_failure_is_a_typed_network_error(self):
        client = MagicMock()
        client.retrieve.side_effect = OSError("proxy denied")
        opendata = types.ModuleType("ecmwf.opendata")
        opendata.Client = MagicMock(return_value=client)
        ecmwf = types.ModuleType("ecmwf")
        ecmwf.opendata = opendata

        with patch.dict(sys.modules, {"ecmwf": ecmwf, "ecmwf.opendata": opendata}), \
             pytest.raises(EcmwfNetworkError, match="proxy denied"):
            fetch_weather_forecast(-75.0, 180.0)


@pytest.mark.integration
def test_fetch_small_live_antarctic_ecmwf_sample():
    """Request one current IFS step and sample it at an Antarctic ocean point."""
    try:
        snapshot = fetch_weather_forecast(lat=-75.0, lon=180.0)
    except EcmwfNetworkError as exc:
        pytest.skip(f"ECMWF Open Data unreachable: UNVERIFIED ({exc})")

    assert isinstance(snapshot, WeatherSnapshot)
    assert snapshot.available_variables == ["u10", "v10", "msl", "swh", "t2m"]
    assert snapshot.valid_at == snapshot.source_data_timestamp
    assert snapshot.metadata["provider"] == "ECMWF IFS Open Data"
    assert snapshot.pressure_hpa is not None and 800.0 <= snapshot.pressure_hpa <= 1100.0
    assert snapshot.significant_wave_height_m is not None and 0.0 <= snapshot.significant_wave_height_m <= 30.0
    assert snapshot.air_temperature_c is not None and -90.0 <= snapshot.air_temperature_c <= 40.0
    assert issubclass(EcmwfNetworkError, WeatherConnectorError)
