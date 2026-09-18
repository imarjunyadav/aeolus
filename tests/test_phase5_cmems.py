"""
Tests for the CMEMS ocean connector (cloud/data/ocean/cmems.py).

Unit tests use unittest.mock to avoid any real network calls.
The single integration test attempts a real tiny subset (Ross Sea, 1 day)
and skips if the proxy blocks the CMEMS hosts.
"""

from __future__ import annotations

import sys
import types
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cloud.data.exceptions import OceanConnectorError
from cloud.data.ocean.cmems import (
    CmemsAuthError,
    CmemsNetworkError,
    CmemsNotFoundError,
    CmemsParseError,
    _classify_cmems_error,
    _ds_to_snapshot,
    _extract_model_time,
    _scalar,
    fetch_ocean_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_TIME = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)


def _make_ds(
    uo: float | None = 0.15,
    vo: float | None = 0.05,
    thetao: float | None = 1.5,
    siconc: float | None = 0.3,
    sithick: float | None = 0.8,
    mlotst: float | None = 50.0,
    dims: tuple[str, ...] = ("time", "latitude", "longitude"),
    model_time: datetime = _MODEL_TIME,
    include_time_coord: bool = True,
) -> MagicMock:
    """
    Build a mock xarray Dataset whose variables return scalars or NaN arrays.

    Includes a ``time`` coordinate (numpy datetime64) matching model_time,
    which _extract_model_time() reads to determine source_data_timestamp.
    Set include_time_coord=False to test the missing-time-coord error path.
    """
    import pandas as pd

    ds = MagicMock()

    def _make_da(value: float | None) -> MagicMock:
        da = MagicMock()
        da.dims = list(dims)
        if value is None:
            arr = np.array([[[np.nan]]])
        else:
            arr = np.array([[[value]]])
        da.values = arr
        # isel returns a da without the dropped dim (simplified)
        da.isel.return_value = da
        return da

    # Mock the time coordinate: values is a numpy datetime64 array
    time_da = MagicMock()
    if include_time_coord:
        time_da.values = np.array([pd.Timestamp(model_time).to_datetime64()])
    else:
        time_da.values = np.array([])  # empty → IndexError in _extract_model_time

    mapping = {
        "uo":      _make_da(uo),
        "vo":      _make_da(vo),
        "thetao":  _make_da(thetao),
        "siconc":  _make_da(siconc),
        "sithick": _make_da(sithick),
        "mlotst":  _make_da(mlotst),
        "time":    time_da,
    }

    def getitem(key: str) -> MagicMock:
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    ds.__getitem__ = MagicMock(side_effect=getitem)
    return ds


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_auth_is_ocean_connector_error(self):
        assert issubclass(CmemsAuthError, OceanConnectorError)

    def test_notfound_is_ocean_connector_error(self):
        assert issubclass(CmemsNotFoundError, OceanConnectorError)

    def test_network_is_ocean_connector_error(self):
        assert issubclass(CmemsNetworkError, OceanConnectorError)

    def test_parse_is_ocean_connector_error(self):
        assert issubclass(CmemsParseError, OceanConnectorError)

    def test_all_distinct(self):
        classes = [CmemsAuthError, CmemsNotFoundError, CmemsNetworkError, CmemsParseError]
        assert len(set(classes)) == len(classes)


# ---------------------------------------------------------------------------
# _classify_cmems_error
# ---------------------------------------------------------------------------

class TestClassifyError:
    def _fake_exc(self, name: str, msg: str = "test") -> Exception:
        """Create an exception whose type.__name__ == name."""
        exc_type = type(name, (Exception,), {})
        return exc_type(msg)

    @pytest.mark.parametrize("cls_name", [
        "CredentialsCannotBeNone",
        "InvalidUsernameOrPassword",
        "CouldNotConnectToAuthenticationSystem",
    ])
    def test_auth_exceptions(self, cls_name: str):
        exc = self._fake_exc(cls_name)
        result = _classify_cmems_error(exc)
        assert isinstance(result, CmemsAuthError)
        assert cls_name in str(result)

    @pytest.mark.parametrize("cls_name", [
        "ProductNotFound",
        "DatasetNotFound",
        "DatasetVersionNotFound",
        "DatasetVersionPartNotFound",
        "VariableDoesNotExistInTheDataset",
    ])
    def test_notfound_exceptions(self, cls_name: str):
        exc = self._fake_exc(cls_name)
        result = _classify_cmems_error(exc)
        assert isinstance(result, CmemsNotFoundError)

    def test_oserror_maps_to_network(self):
        exc = OSError("connection refused")
        result = _classify_cmems_error(exc)
        assert isinstance(result, CmemsNetworkError)

    def test_connection_error_maps_to_network(self):
        exc = ConnectionError("failed")
        result = _classify_cmems_error(exc)
        assert isinstance(result, CmemsNetworkError)

    def test_unknown_exception_maps_to_network(self):
        exc = self._fake_exc("SomeUnknownCmemsError", "oops")
        result = _classify_cmems_error(exc)
        assert isinstance(result, CmemsNetworkError)
        assert "SomeUnknownCmemsError" in str(result)


# ---------------------------------------------------------------------------
# _scalar
# ---------------------------------------------------------------------------

class TestScalar:
    def test_normal_value(self):
        ds = _make_ds(uo=0.25)
        result = _scalar(ds, "uo")
        assert result == pytest.approx(0.25)

    def test_none_for_allnan(self):
        ds = _make_ds(uo=None)
        result = _scalar(ds, "uo")
        assert result is None

    def test_missing_variable(self):
        ds = _make_ds()
        # Override the side_effect so only "missing_var" raises KeyError
        ds.__getitem__ = MagicMock(side_effect=KeyError("missing_var"))
        result = _scalar(ds, "missing_var")
        assert result is None

    def test_negative_value(self):
        ds = _make_ds(vo=-0.12)
        result = _scalar(ds, "vo")
        assert result == pytest.approx(-0.12)

    def test_zero_value(self):
        ds = _make_ds(uo=0.0)
        result = _scalar(ds, "uo")
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _ds_to_snapshot
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _extract_model_time
# ---------------------------------------------------------------------------

class TestExtractModelTime:
    def test_returns_utc_datetime_from_numpy_ts(self):
        import pandas as pd
        model_dt = datetime(2026, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
        ds = _make_ds(model_time=model_dt)
        result = _extract_model_time(ds)
        assert result is not None
        assert result == model_dt

    def test_returns_none_when_time_coord_missing(self):
        ds = _make_ds()
        # Remove "time" from the mapping so KeyError is raised
        ds.__getitem__ = MagicMock(side_effect=KeyError("time"))
        result = _extract_model_time(ds)
        assert result is None

    def test_returns_none_when_time_array_empty(self):
        ds = _make_ds(include_time_coord=False)
        result = _extract_model_time(ds)
        assert result is None

    def test_result_is_timezone_aware(self):
        ds = _make_ds()
        result = _extract_model_time(ds)
        assert result is not None
        assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# _ds_to_snapshot
# ---------------------------------------------------------------------------

class TestDsToSnapshot:
    # fetch_time is 1 hour after model_time → age = 3600 s
    _FETCH_TIME = datetime(2026, 9, 14, 13, 0, 0, tzinfo=timezone.utc)

    def test_all_variables_present(self):
        ds = _make_ds(uo=0.15, vo=-0.05, thetao=1.5,
                      siconc=0.3, sithick=0.8, mlotst=50.0)
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        assert snap.current_u_ms == pytest.approx(0.15)
        assert snap.current_v_ms == pytest.approx(-0.05)
        assert snap.sst_c == pytest.approx(1.5)
        assert "uo" in snap.available_variables
        assert "vo" in snap.available_variables
        assert "thetao" in snap.available_variables

    def test_metadata_contains_product_id(self):
        ds = _make_ds()
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        assert "GLOBAL_ANALYSISFORECAST_PHY_001_024" in snap.metadata["product_id"]

    def test_metadata_contains_dataset_id(self):
        ds = _make_ds()
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        assert "cmems_mod_glo_phy_anfc_0.083deg_P1D-m" in snap.metadata["dataset_id"]

    def test_metadata_contains_fetch_time(self):
        ds = _make_ds()
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        assert "fetch_time_utc" in snap.metadata
        assert "2026-09-14T13:00:00" in snap.metadata["fetch_time_utc"]

    def test_optional_fields_in_metadata(self):
        ds = _make_ds(siconc=0.5, sithick=1.2, mlotst=35.0)
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        assert snap.metadata["siconc"] == pytest.approx(0.5)
        assert snap.metadata["sithick"] == pytest.approx(1.2)
        assert snap.metadata["mlotst"] == pytest.approx(35.0)

    def test_raises_parse_error_when_both_currents_nan(self):
        ds = _make_ds(uo=None, vo=None)
        with pytest.raises(CmemsParseError, match="uo and vo"):
            _ds_to_snapshot(ds, self._FETCH_TIME)

    def test_raises_parse_error_when_time_coord_missing(self):
        ds = _make_ds()
        ds.__getitem__ = MagicMock(side_effect=KeyError("time"))
        with pytest.raises(CmemsParseError, match="time"):
            _ds_to_snapshot(ds, self._FETCH_TIME)

    def test_partial_availability(self):
        # sithick missing — should still succeed if uo/vo present
        ds = _make_ds(uo=0.1, vo=0.0, thetao=2.0, sithick=None)
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        assert "sithick" not in snap.available_variables
        assert snap.metadata["sithick"] is None

    def test_source_timestamp_is_model_time_not_fetch_time(self):
        """source_data_timestamp must be the CMEMS model time, not the download time."""
        ds = _make_ds(model_time=_MODEL_TIME)
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        assert snap.source_data_timestamp == _MODEL_TIME
        assert snap.valid_at == _MODEL_TIME
        # fetch_time is distinct from model time
        assert snap.source_data_timestamp != self._FETCH_TIME

    def test_source_data_age_is_fetch_minus_model_time(self):
        """source_data_age_seconds = fetch_time − model_time in seconds."""
        # _MODEL_TIME = 2026-09-13 12:00 UTC, _FETCH_TIME = 2026-09-14 13:00 UTC
        # delta = 25 hours = 90000 seconds
        ds = _make_ds(model_time=_MODEL_TIME)
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        expected_age = int((self._FETCH_TIME - _MODEL_TIME).total_seconds())
        assert snap.source_data_age_seconds == expected_age

    def test_source_data_age_not_hardcoded_zero(self):
        ds = _make_ds()
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        assert snap.source_data_age_seconds > 0

    def test_available_variables_list_correct(self):
        ds = _make_ds(uo=0.1, vo=0.2, thetao=1.0, siconc=0.4,
                      sithick=None, mlotst=None)
        snap = _ds_to_snapshot(ds, self._FETCH_TIME)
        assert set(snap.available_variables) == {"uo", "vo", "thetao", "siconc"}


# ---------------------------------------------------------------------------
# fetch_ocean_snapshot — unit (mocked copernicusmarine)
# ---------------------------------------------------------------------------

class TestFetchOceanSnapshotUnit:
    """Mock copernicusmarine.open_dataset to avoid any real network calls."""

    def _mock_cm(self, ds_return: MagicMock) -> MagicMock:
        cm = MagicMock()
        cm.open_dataset.return_value = ds_return
        return cm

    def test_calls_open_dataset_with_correct_params(self):
        ds = _make_ds()
        cm = self._mock_cm(ds)

        with patch.dict("sys.modules", {"copernicusmarine": cm}):
            snap = fetch_ocean_snapshot(
                lat_min=-80.0, lat_max=-75.0,
                lon_min=-180.0, lon_max=-150.0,
                target_date=date(2026, 9, 14),
            )

        call_kwargs = cm.open_dataset.call_args.kwargs
        assert call_kwargs["dataset_id"] == "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
        assert call_kwargs["minimum_latitude"] == -80.0
        assert call_kwargs["maximum_latitude"] == -75.0
        assert call_kwargs["minimum_longitude"] == -180.0
        assert call_kwargs["maximum_longitude"] == -150.0
        assert "uo" in call_kwargs["variables"]
        assert "vo" in call_kwargs["variables"]

    def test_returns_ocean_snapshot(self):
        ds = _make_ds(uo=0.2, vo=-0.03, thetao=0.5)
        cm = self._mock_cm(ds)

        with patch.dict("sys.modules", {"copernicusmarine": cm}):
            snap = fetch_ocean_snapshot(
                lat_min=-80.0, lat_max=-75.0,
                lon_min=-180.0, lon_max=-150.0,
                target_date=date(2026, 9, 14),
            )

        assert snap.current_u_ms == pytest.approx(0.2)
        assert snap.current_v_ms == pytest.approx(-0.03)
        assert snap.sst_c == pytest.approx(0.5)

    def test_explicit_credentials_forwarded(self):
        ds = _make_ds()
        cm = self._mock_cm(ds)

        with patch.dict("sys.modules", {"copernicusmarine": cm}):
            fetch_ocean_snapshot(
                lat_min=-80.0, lat_max=-75.0,
                lon_min=-180.0, lon_max=-150.0,
                target_date=date(2026, 9, 14),
                username="myuser",
                password="mypass",
            )

        kwargs = cm.open_dataset.call_args.kwargs
        assert kwargs["username"] == "myuser"
        assert kwargs["password"] == "mypass"

    def test_no_credentials_kwarg_when_not_provided(self):
        ds = _make_ds()
        cm = self._mock_cm(ds)

        with patch.dict("sys.modules", {"copernicusmarine": cm}):
            fetch_ocean_snapshot(
                lat_min=-80.0, lat_max=-75.0,
                lon_min=-180.0, lon_max=-150.0,
                target_date=date(2026, 9, 14),
            )

        kwargs = cm.open_dataset.call_args.kwargs
        assert "username" not in kwargs
        assert "password" not in kwargs

    @pytest.mark.parametrize("exc_cls_name,expected", [
        ("CredentialsCannotBeNone", CmemsAuthError),
        ("InvalidUsernameOrPassword", CmemsAuthError),
        ("CouldNotConnectToAuthenticationSystem", CmemsAuthError),
        ("ProductNotFound", CmemsNotFoundError),
        ("DatasetNotFound", CmemsNotFoundError),
    ])
    def test_open_dataset_errors_classified(self, exc_cls_name: str, expected: type):
        exc_type = type(exc_cls_name, (Exception,), {})

        cm = MagicMock()
        cm.open_dataset.side_effect = exc_type("error")

        with patch.dict("sys.modules", {"copernicusmarine": cm}):
            with pytest.raises(expected):
                fetch_ocean_snapshot(
                    lat_min=-80.0, lat_max=-75.0,
                    lon_min=-180.0, lon_max=-150.0,
                    target_date=date(2026, 9, 14),
                )

    def test_oserror_becomes_network_error(self):
        cm = MagicMock()
        cm.open_dataset.side_effect = OSError("proxy error")

        with patch.dict("sys.modules", {"copernicusmarine": cm}):
            with pytest.raises(CmemsNetworkError):
                fetch_ocean_snapshot(
                    lat_min=-80.0, lat_max=-75.0,
                    lon_min=-180.0, lon_max=-150.0,
                    target_date=date(2026, 9, 14),
                )

    def test_parse_error_propagated(self):
        # uo and vo both NaN → CmemsParseError
        ds = _make_ds(uo=None, vo=None)
        cm = self._mock_cm(ds)

        with patch.dict("sys.modules", {"copernicusmarine": cm}):
            with pytest.raises(CmemsParseError):
                fetch_ocean_snapshot(
                    lat_min=-80.0, lat_max=-75.0,
                    lon_min=-180.0, lon_max=-150.0,
                    target_date=date(2026, 9, 14),
                )

    def test_import_error_raises_network_error(self):
        """If copernicusmarine cannot be imported, raises CmemsNetworkError."""
        # Setting sys.modules entry to None makes `import copernicusmarine` raise ImportError.
        with patch.dict("sys.modules", {"copernicusmarine": None}):
            with pytest.raises(CmemsNetworkError, match="not installed"):
                fetch_ocean_snapshot(
                    lat_min=-80.0, lat_max=-75.0,
                    lon_min=-180.0, lon_max=-150.0,
                    target_date=date(2026, 9, 14),
                )

    def test_default_date_is_yesterday(self):
        """When target_date is None, yesterday UTC is used."""
        ds = _make_ds()
        cm = self._mock_cm(ds)

        with patch.dict("sys.modules", {"copernicusmarine": cm}):
            fetch_ocean_snapshot(
                lat_min=-80.0, lat_max=-75.0,
                lon_min=-180.0, lon_max=-150.0,
            )

        kwargs = cm.open_dataset.call_args.kwargs
        yesterday = (datetime.now(tz=timezone.utc) - __import__("datetime").timedelta(days=1)).date()
        start: datetime = kwargs["start_datetime"]
        assert start.date() == yesterday


# ---------------------------------------------------------------------------
# Integration test — real CMEMS access, tiny Ross Sea subset
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCmemsIntegration:
    """
    Live integration test against real CMEMS.

    Fetches a tiny 5°×5° Ross Sea box for one day.
    Skipped automatically when:
      - copernicusmarine cannot connect (proxy 403)
      - CMEMS credentials are absent from the environment
    """

    # Ross Sea corner — deep Southern Ocean, always has ocean data
    LAT_MIN, LAT_MAX = -80.0, -75.0
    LON_MIN, LON_MAX = -180.0, -175.0
    TARGET_DATE = date(2026, 9, 13)  # yesterday-ish

    def test_fetch_ross_sea_snapshot(self):
        try:
            snap = fetch_ocean_snapshot(
                lat_min=self.LAT_MIN,
                lat_max=self.LAT_MAX,
                lon_min=self.LON_MIN,
                lon_max=self.LON_MAX,
                target_date=self.TARGET_DATE,
            )
        except CmemsAuthError as exc:
            pytest.skip(f"CMEMS credentials absent or rejected: {exc}")
        except CmemsNetworkError as exc:
            err = str(exc)
            _skip_tokens = ("403", "forbidden", "proxy", "stdin",
                            "credentials", "username", "password")
            if any(t in err.lower() for t in _skip_tokens):
                pytest.skip(f"CMEMS unreachable or credentials absent: {exc}")
            raise  # unexpected network failure — let it fail the test

        # Structural assertions on the returned snapshot
        from shared.schemas.forecast import OceanSnapshot
        assert isinstance(snap, OceanSnapshot)

        # At least one current component should be present in the Ross Sea
        assert snap.current_u_ms is not None or snap.current_v_ms is not None, (
            "Expected at least one current component (uo/vo) to be non-None"
        )

        # Values must be physically plausible for the Southern Ocean
        if snap.current_u_ms is not None:
            assert -3.0 < snap.current_u_ms < 3.0, f"uo out of range: {snap.current_u_ms}"
        if snap.current_v_ms is not None:
            assert -3.0 < snap.current_v_ms < 3.0, f"vo out of range: {snap.current_v_ms}"
        if snap.sst_c is not None:
            assert -5.0 < snap.sst_c < 30.0, f"thetao out of range: {snap.sst_c}"
        if snap.metadata.get("siconc") is not None:
            assert 0.0 <= snap.metadata["siconc"] <= 1.0, (
                f"siconc out of range: {snap.metadata['siconc']}"
            )

        assert snap.metadata["product_id"] == "GLOBAL_ANALYSISFORECAST_PHY_001_024"
        assert "cmems_mod_glo_phy_anfc_0.083deg_P1D-m" in snap.metadata["dataset_id"]

        print(f"\n[INTEGRATION] CMEMS Ross Sea snapshot for {self.TARGET_DATE}:")
        print(f"  uo={snap.current_u_ms:.4f} m/s" if snap.current_u_ms else "  uo=None")
        print(f"  vo={snap.current_v_ms:.4f} m/s" if snap.current_v_ms else "  vo=None")
        print(f"  thetao={snap.sst_c:.2f} °C" if snap.sst_c else "  thetao=None")
        print(f"  siconc={snap.metadata.get('siconc')}")
        print(f"  sithick={snap.metadata.get('sithick')}")
        print(f"  mlotst={snap.metadata.get('mlotst')}")
        print(f"  available_variables={snap.available_variables}")
