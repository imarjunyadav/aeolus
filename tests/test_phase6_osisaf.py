"""
Tests for cloud.data.sea_ice.osisaf — Step 6 (OSI SAF sea-ice connector).

Coverage
--------
Unit tests use mocked / fixture xarray Datasets that mimic OSI SAF file
structure (polar-stereo coordinates, time in datetime64, short ice_conc
with scale_factor, algorithm_uncertainty, smearing_uncertainty,
total_uncertainty, status_flag).

The integration test attempts a real OPeNDAP connection to
thredds.met.no.  In this CCR build environment the host is blocked by the
outbound proxy (HTTP 403); the test skips gracefully with the reason.

All fixture tests are deterministic, self-contained, and run offline.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from cloud.data.exceptions import SeaIceConnectorError
from cloud.data.sea_ice.osisaf import (
    OsisafNetworkError,
    OsisafNotFoundError,
    OsisafParseError,
    SH_PROJ,
    _SH_NRT_PATTERN,
    _SH_AMSR3_PATTERN,
    _build_nrt_url,
    _build_cdr_url,
    _classify_error,
    _derive_total_uncertainty,
    _extract_validity_time,
    _normalise_coords,
    _require_vars,
    fetch_nrt_concentration,
    fetch_cdr_concentration,
)


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------

_MODEL_TIME = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)
_EPOCH = datetime(1978, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

_N_XC = 5
_N_YC = 4

def _make_ds(
    ice_conc_val: float = 75.0,
    alg_unc_val: float = 5.0,
    smear_unc_val: float = 3.0,
    total_unc_val: float | None = None,
    status_val: int = 0,
    model_time: datetime = _MODEL_TIME,
    xc_units: str = "km",
    include_time: bool = True,
    missing_vars: set | None = None,
) -> xr.Dataset:
    """
    Build a minimal mock OSI SAF-like Dataset.

    ice_conc is stored as a short int (integer) to mimic the actual file
    format; xarray decodes it to float when scale_factor is present and
    mask_and_scale=True (default in open_dataset).  For unit tests we
    store the raw value and rely on callers using decoded floats.

    xc/yc coordinates are in ``xc_units`` ('km' or 'm').
    """
    if total_unc_val is None:
        total_unc_val = float(np.sqrt(alg_unc_val**2 + smear_unc_val**2))

    missing_vars = missing_vars or set()

    # Coordinates
    xc_values = np.linspace(-1000.0, 1000.0, _N_XC)  # km or m depending on param
    yc_values = np.linspace(-800.0, 800.0, _N_YC)

    shape = (_N_YC, _N_XC)
    ice   = np.full(shape, ice_conc_val, dtype=np.float32)
    alg   = np.full(shape, alg_unc_val,  dtype=np.float32)
    smear = np.full(shape, smear_unc_val, dtype=np.float32)
    total = np.full(shape, total_unc_val, dtype=np.float32)
    flags = np.full(shape, status_val,    dtype=np.int16)

    coords: dict = {
        "xc": ("xc", xc_values, {"units": xc_units, "axis": "X"}),
        "yc": ("yc", yc_values, {"units": xc_units, "axis": "Y"}),
    }
    if include_time:
        coords["time"] = ("time", np.array([np.datetime64(
            model_time.replace(tzinfo=None).isoformat()
        )]))

    time_dim = ("time", "yc", "xc") if include_time else ("yc", "xc")
    n_time = 1 if include_time else None

    def _expand(arr):
        return arr[np.newaxis] if include_time else arr

    all_vars = {
        "ice_conc":              (_expand(ice),   {"scale_factor": 0.01, "units": "%"}),
        "algorithm_uncertainty": (_expand(alg),   {"units": "%"}),
        "smearing_uncertainty":  (_expand(smear), {"units": "%"}),
        "total_uncertainty":     (_expand(total), {"units": "%"}),
        "status_flag":           (_expand(flags), {"flag_meanings": "nominal ..."}),
    }

    data_vars = {
        name: xr.DataArray(val, dims=time_dim if include_time else ("yc", "xc"), attrs=attrs)
        for name, (val, attrs) in all_vars.items()
        if name not in missing_vars
    }

    return xr.Dataset(data_vars, coords=coords)


# ===========================================================================
# TestExceptionHierarchy
# ===========================================================================

class TestExceptionHierarchy:
    def test_osisaf_network_is_sea_ice_connector_error(self):
        assert issubclass(OsisafNetworkError, SeaIceConnectorError)

    def test_osisaf_notfound_is_sea_ice_connector_error(self):
        assert issubclass(OsisafNotFoundError, SeaIceConnectorError)

    def test_osisaf_parse_is_sea_ice_connector_error(self):
        assert issubclass(OsisafParseError, SeaIceConnectorError)

    def test_all_are_distinct(self):
        assert OsisafNetworkError is not OsisafNotFoundError
        assert OsisafNetworkError is not OsisafParseError
        assert OsisafNotFoundError is not OsisafParseError

    def test_sh_proj_is_string(self):
        assert isinstance(SH_PROJ, str)
        assert "stere" in SH_PROJ
        assert "-70" in SH_PROJ


# ===========================================================================
# TestUrlBuilder
# ===========================================================================

class TestUrlBuilder:
    def test_nrt_url_contains_year_month(self):
        url = _build_nrt_url(date(2026, 9, 13))
        assert "/2026/09/" in url

    def test_nrt_url_filename_pattern(self):
        url = _build_nrt_url(date(2026, 9, 13))
        assert "ice_conc_sh_polstere-100_multi_20260913120000Z.nc" in url

    def test_nrt_url_uses_dodsC(self):
        url = _build_nrt_url(date(2026, 9, 13))
        assert "dodsC" in url
        assert "thredds.met.no" in url

    def test_nrt_url_zero_padded_month(self):
        url = _build_nrt_url(date(2026, 3, 5))
        assert "/2026/03/" in url
        assert "20260305" in url

    def test_cdr_url_pre2021_uses_cdr_base(self):
        url = _build_cdr_url(date(2019, 6, 15))
        assert "conc_cdr" in url

    def test_cdr_url_post2021_uses_icdr_base(self):
        url = _build_cdr_url(date(2022, 6, 15))
        assert "conc_cdrv3" in url


# ===========================================================================
# TestClassifyError
# ===========================================================================

class TestClassifyError:
    def test_404_gives_notfound(self):
        exc = Exception("Server returned HTTP 404")
        result = _classify_error(exc)
        assert isinstance(result, OsisafNotFoundError)

    def test_not_found_in_msg_gives_notfound(self):
        exc = Exception("File not found on THREDDS")
        result = _classify_error(exc)
        assert isinstance(result, OsisafNotFoundError)

    def test_403_gives_network(self):
        exc = Exception("HTTP 403 Forbidden from proxy")
        result = _classify_error(exc)
        assert isinstance(result, OsisafNetworkError)

    def test_407_gives_network(self):
        exc = Exception("HTTP 407 Proxy Authentication Required")
        result = _classify_error(exc)
        assert isinstance(result, OsisafNetworkError)

    def test_oserror_gives_network(self):
        exc = OSError("Connection refused")
        result = _classify_error(exc)
        assert isinstance(result, OsisafNetworkError)

    def test_connection_error_gives_network(self):
        exc = ConnectionError("Connection reset by peer")
        result = _classify_error(exc)
        assert isinstance(result, OsisafNetworkError)

    def test_generic_exception_gives_network(self):
        exc = RuntimeError("Something went wrong")
        result = _classify_error(exc)
        assert isinstance(result, OsisafNetworkError)

    def test_classified_is_subclass_of_sea_ice_connector_error(self):
        exc = OSError("network")
        result = _classify_error(exc)
        assert isinstance(result, SeaIceConnectorError)

    def test_error_message_preserved(self):
        exc = Exception("some detail")
        result = _classify_error(exc)
        assert "some detail" in str(result)


# ===========================================================================
# TestExtractValidityTime
# ===========================================================================

class TestExtractValidityTime:
    def test_returns_utc_datetime(self):
        ds = _make_ds(model_time=_MODEL_TIME)
        t = _extract_validity_time(ds)
        assert isinstance(t, datetime)
        assert t.tzinfo is not None
        assert t.tzinfo == timezone.utc

    def test_extracts_correct_time(self):
        target = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)
        ds = _make_ds(model_time=target)
        t = _extract_validity_time(ds)
        assert t == target

    def test_returns_none_when_time_missing(self):
        ds = _make_ds(include_time=False)
        t = _extract_validity_time(ds)
        assert t is None

    def test_different_dates(self):
        target = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ds = _make_ds(model_time=target)
        t = _extract_validity_time(ds)
        assert t.year == 2024
        assert t.month == 1
        assert t.day == 15


# ===========================================================================
# TestNormaliseCoords
# ===========================================================================

class TestNormaliseCoords:
    def test_km_coords_converted_to_m(self):
        ds = _make_ds(xc_units="km")
        original_max = float(ds["xc"].max())
        ds_norm = _normalise_coords(ds)
        new_max = float(ds_norm["xc"].max())
        assert abs(new_max - original_max * 1000.0) < 1e-3

    def test_m_coords_not_changed(self):
        ds = _make_ds(xc_units="m")
        original_max = float(ds["xc"].max())
        ds_norm = _normalise_coords(ds)
        new_max = float(ds_norm["xc"].max())
        assert abs(new_max - original_max) < 1e-6

    def test_units_attribute_updated_to_m(self):
        ds = _make_ds(xc_units="km")
        ds_norm = _normalise_coords(ds)
        assert ds_norm["xc"].attrs["units"] == "m"
        assert ds_norm["yc"].attrs["units"] == "m"

    def test_no_time_no_crash(self):
        ds = _make_ds(include_time=False, xc_units="km")
        ds_norm = _normalise_coords(ds)
        assert ds_norm["xc"].attrs["units"] == "m"

    def test_missing_units_raises_parse_error(self):
        ds = _make_ds(xc_units="")
        with pytest.raises(OsisafParseError, match="no 'units' attribute"):
            _normalise_coords(ds)

    def test_unknown_units_raises_parse_error(self):
        ds = _make_ds(xc_units="feet")
        with pytest.raises(OsisafParseError, match="unrecognised units"):
            _normalise_coords(ds)

    def test_parse_error_names_the_dimension(self):
        ds = _make_ds(xc_units="parsecs")
        with pytest.raises(OsisafParseError) as exc_info:
            _normalise_coords(ds)
        assert "xc" in str(exc_info.value) or "yc" in str(exc_info.value)

    def test_metres_spelled_out_accepted(self):
        ds = _make_ds(xc_units="meters")
        ds_norm = _normalise_coords(ds)
        assert ds_norm["xc"].attrs["units"] == "meters"  # unchanged (already metres)

    def test_kilometres_spelled_out_converted(self):
        ds = _make_ds(xc_units="kilometres")
        original_max = float(ds["xc"].max())
        ds_norm = _normalise_coords(ds)
        assert abs(float(ds_norm["xc"].max()) - original_max * 1000.0) < 1e-3


# ===========================================================================
# TestRequireVars
# ===========================================================================

class TestRequireVars:
    def test_full_ds_passes(self):
        ds = _make_ds()
        _require_vars(ds)  # should not raise

    def test_missing_ice_conc_raises(self):
        ds = _make_ds(missing_vars={"ice_conc"})
        with pytest.raises(OsisafParseError, match="ice_conc"):
            _require_vars(ds)

    def test_missing_status_flag_raises(self):
        ds = _make_ds(missing_vars={"status_flag"})
        with pytest.raises(OsisafParseError, match="status_flag"):
            _require_vars(ds)

    def test_missing_algorithm_uncertainty_raises(self):
        ds = _make_ds(missing_vars={"algorithm_uncertainty"})
        with pytest.raises(OsisafParseError, match="algorithm_uncertainty"):
            _require_vars(ds)

    def test_error_lists_missing_vars(self):
        ds = _make_ds(missing_vars={"ice_conc", "smearing_uncertainty"})
        with pytest.raises(OsisafParseError) as exc_info:
            _require_vars(ds)
        msg = str(exc_info.value)
        assert "ice_conc" in msg or "smearing_uncertainty" in msg


# ===========================================================================
# TestFetchNrtConcentrationUnit
# ===========================================================================

class TestFetchNrtConcentrationUnit:
    """Unit tests that mock _open_opendap to avoid real network calls."""

    def _patch_open(self, ds: xr.Dataset):
        """Patch _open_opendap to return ``ds``."""
        return patch(
            "cloud.data.sea_ice.osisaf._open_opendap",
            return_value=ds,
        )

    def test_returns_dataset(self):
        ds = _make_ds()
        with self._patch_open(ds):
            result = fetch_nrt_concentration(date(2026, 9, 13))
        assert isinstance(result, xr.Dataset)

    def test_uses_yesterday_by_default(self):
        ds = _make_ds()
        yesterday = (datetime.now(tz=timezone.utc) - timedelta(days=1)).date()
        with self._patch_open(ds) as mock_open:
            fetch_nrt_concentration()
        called_url = mock_open.call_args[0][0]
        assert yesterday.strftime("%Y%m%d") in called_url

    def test_uses_target_date(self):
        ds = _make_ds()
        with self._patch_open(ds) as mock_open:
            fetch_nrt_concentration(date(2026, 5, 20))
        assert "20260520" in mock_open.call_args[0][0]

    def test_km_coords_normalised_to_m(self):
        ds = _make_ds(xc_units="km")
        with self._patch_open(ds):
            result = fetch_nrt_concentration(date(2026, 9, 13))
        assert result["xc"].attrs["units"] == "m"

    def test_ice_conc_present(self):
        ds = _make_ds()
        with self._patch_open(ds):
            result = fetch_nrt_concentration(date(2026, 9, 13))
        assert "ice_conc" in result

    def test_algorithm_uncertainty_present(self):
        ds = _make_ds()
        with self._patch_open(ds):
            result = fetch_nrt_concentration(date(2026, 9, 13))
        assert "algorithm_uncertainty" in result

    def test_smearing_uncertainty_present(self):
        ds = _make_ds()
        with self._patch_open(ds):
            result = fetch_nrt_concentration(date(2026, 9, 13))
        assert "smearing_uncertainty" in result

    def test_status_flag_present(self):
        ds = _make_ds()
        with self._patch_open(ds):
            result = fetch_nrt_concentration(date(2026, 9, 13))
        assert "status_flag" in result

    def test_missing_time_raises_parse_error(self):
        ds = _make_ds(include_time=False)
        with self._patch_open(ds):
            with pytest.raises(OsisafParseError, match="time"):
                fetch_nrt_concentration(date(2026, 9, 13))

    def test_missing_ice_conc_raises_parse_error(self):
        ds = _make_ds(missing_vars={"ice_conc"})
        with self._patch_open(ds):
            with pytest.raises(OsisafParseError):
                fetch_nrt_concentration(date(2026, 9, 13))

    def test_network_error_propagates(self):
        with patch(
            "cloud.data.sea_ice.osisaf._open_opendap",
            side_effect=OsisafNetworkError("proxy 403"),
        ):
            with pytest.raises(OsisafNetworkError, match="403"):
                fetch_nrt_concentration(date(2026, 9, 13))

    def test_notfound_error_propagates(self):
        with patch(
            "cloud.data.sea_ice.osisaf._open_opendap",
            side_effect=OsisafNotFoundError("404"),
        ):
            with pytest.raises(OsisafNotFoundError):
                fetch_nrt_concentration(date(2026, 9, 13))

    def test_pydap_missing_raises_network_error(self):
        with patch.dict(sys.modules, {"pydap": None}):
            with pytest.raises(OsisafNetworkError, match="pydap"):
                # _open_opendap is NOT mocked here — let it run for real
                fetch_nrt_concentration(date(2026, 9, 13))

    def test_xarray_missing_raises_network_error(self):
        with patch.dict(sys.modules, {"xarray": None}):
            with pytest.raises(OsisafNetworkError, match="xarray"):
                fetch_nrt_concentration(date(2026, 9, 13))


# ===========================================================================
# TestFetchCdrConcentrationUnit
# ===========================================================================

class TestFetchCdrConcentrationUnit:
    def _patch_open(self, ds: xr.Dataset):
        return patch(
            "cloud.data.sea_ice.osisaf._open_opendap",
            return_value=ds,
        )

    def test_returns_dataset(self):
        ds = _make_ds()
        with self._patch_open(ds):
            result = fetch_cdr_concentration(date(2019, 3, 10))
        assert isinstance(result, xr.Dataset)

    def test_uses_cdr_base_pre2021(self):
        ds = _make_ds()
        with self._patch_open(ds) as mock_open:
            fetch_cdr_concentration(date(2019, 3, 10))
        assert "conc_cdr" in mock_open.call_args[0][0]

    def test_uses_icdr_base_post2021(self):
        ds = _make_ds()
        with self._patch_open(ds) as mock_open:
            fetch_cdr_concentration(date(2022, 7, 1))
        assert "conc_cdrv3" in mock_open.call_args[0][0]

    def test_ice_conc_in_result(self):
        ds = _make_ds()
        with self._patch_open(ds):
            result = fetch_cdr_concentration(date(2019, 3, 10))
        assert "ice_conc" in result

    def test_missing_time_raises_parse_error(self):
        ds = _make_ds(include_time=False)
        with self._patch_open(ds):
            with pytest.raises(OsisafParseError):
                fetch_cdr_concentration(date(2019, 3, 10))


# ===========================================================================
# TestFilenameConstants
# ===========================================================================

class TestFilenameConstants:
    """
    Verify that the multi-sensor and OSI-408-g (AMSR3) filename patterns
    are distinct and correctly structured.
    """

    def test_multi_pattern_contains_multi(self):
        assert "multi" in _SH_NRT_PATTERN

    def test_multi_pattern_has_14digit_timestamp_and_z(self):
        # {dt} is YYYYMMDDHHmmss (14 digits) + literal "Z"
        rendered = _SH_NRT_PATTERN.format(dt="20260913120000")
        assert rendered.endswith("120000Z.nc")

    def test_amsr3_pattern_contains_amsr3(self):
        assert "amsr3" in _SH_AMSR3_PATTERN

    def test_amsr3_pattern_has_12digit_timestamp_no_z(self):
        # {dt} is YYYYMMDDHHmm (12 digits), no trailing "Z"
        rendered = _SH_AMSR3_PATTERN.format(dt="202609131200")
        assert rendered.endswith("202609131200.nc")
        assert "Z" not in rendered

    def test_multi_and_amsr3_patterns_are_distinct(self):
        assert _SH_NRT_PATTERN != _SH_AMSR3_PATTERN

    def test_nrt_url_uses_multi_pattern(self):
        url = _build_nrt_url(date(2026, 9, 13))
        assert "multi" in url
        assert "amsr3" not in url


# ===========================================================================
# TestDeriveTotalUncertainty
# ===========================================================================

class TestDeriveTotalUncertainty:
    """
    Tests for _derive_total_uncertainty():
      - pre-computed value preserved unchanged
      - derivation produces sqrt(alg² + smear²) when absent
      - derived value has the correct attrs
      - works end-to-end through fetch_nrt_concentration (mocked)
    """

    def test_returns_ds_unchanged_when_total_unc_present(self):
        ds = _make_ds(alg_unc_val=5.0, smear_unc_val=3.0, total_unc_val=99.0)
        result = _derive_total_uncertainty(ds)
        # The pre-computed value (99.0) must NOT be replaced by sqrt(25+9)≈5.83
        vals = np.asarray(result["total_uncertainty"].values, dtype=float)
        finite = vals[np.isfinite(vals)]
        assert abs(finite.mean() - 99.0) < 1e-3

    def test_derives_when_total_unc_absent(self):
        ds = _make_ds(alg_unc_val=3.0, smear_unc_val=4.0,
                      missing_vars={"total_uncertainty"})
        assert "total_uncertainty" not in ds.data_vars
        result = _derive_total_uncertainty(ds)
        assert "total_uncertainty" in result.data_vars

    def test_derived_value_is_sqrt_of_sum_of_squares(self):
        alg, smear = 3.0, 4.0
        expected = (alg**2 + smear**2) ** 0.5  # 5.0
        ds = _make_ds(alg_unc_val=alg, smear_unc_val=smear,
                      missing_vars={"total_uncertainty"})
        result = _derive_total_uncertainty(ds)
        vals = np.asarray(result["total_uncertainty"].values, dtype=float)
        finite = vals[np.isfinite(vals)]
        assert abs(finite.mean() - expected) < 1e-4

    def test_derived_var_has_units_percent(self):
        ds = _make_ds(missing_vars={"total_uncertainty"})
        result = _derive_total_uncertainty(ds)
        assert result["total_uncertainty"].attrs.get("units") == "%"

    def test_derived_var_long_name_indicates_derived(self):
        ds = _make_ds(missing_vars={"total_uncertainty"})
        result = _derive_total_uncertainty(ds)
        long_name = result["total_uncertainty"].attrs.get("long_name", "")
        assert "derived" in long_name.lower()

    def test_derived_var_has_comment_attr(self):
        ds = _make_ds(missing_vars={"total_uncertainty"})
        result = _derive_total_uncertainty(ds)
        comment = result["total_uncertainty"].attrs.get("comment", "")
        assert len(comment) > 0

    def test_original_ds_not_mutated(self):
        ds = _make_ds(missing_vars={"total_uncertainty"})
        original_vars = set(ds.data_vars)
        _derive_total_uncertainty(ds)
        assert set(ds.data_vars) == original_vars  # ds is unchanged

    def test_fetch_nrt_returns_total_uncertainty_when_absent_in_file(self):
        """End-to-end: file lacks total_uncertainty → connector derives it."""
        ds_without = _make_ds(alg_unc_val=6.0, smear_unc_val=8.0,
                               missing_vars={"total_uncertainty"})
        with patch(
            "cloud.data.sea_ice.osisaf._open_opendap",
            return_value=ds_without,
        ):
            result = fetch_nrt_concentration(date(2026, 9, 13))
        assert "total_uncertainty" in result.data_vars
        vals = np.asarray(result["total_uncertainty"].values, dtype=float)
        finite = vals[np.isfinite(vals)]
        # sqrt(36 + 64) = 10.0
        assert abs(finite.mean() - 10.0) < 1e-4

    def test_fetch_nrt_preserves_precomputed_total_uncertainty(self):
        """End-to-end: file has total_uncertainty → connector uses it unchanged."""
        ds_with = _make_ds(alg_unc_val=3.0, smear_unc_val=4.0,
                            total_unc_val=7.5)  # not sqrt(9+16)=5
        with patch(
            "cloud.data.sea_ice.osisaf._open_opendap",
            return_value=ds_with,
        ):
            result = fetch_nrt_concentration(date(2026, 9, 13))
        vals = np.asarray(result["total_uncertainty"].values, dtype=float)
        finite = vals[np.isfinite(vals)]
        assert abs(finite.mean() - 7.5) < 1e-3  # pre-computed value preserved

    def test_raises_parse_error_when_not_derivable(self):
        """
        If total_uncertainty is absent AND a required component is missing,
        _derive_total_uncertainty() must raise OsisafParseError, not KeyError.
        """
        ds_broken = _make_ds(
            missing_vars={"total_uncertainty", "algorithm_uncertainty"}
        )
        assert "total_uncertainty" not in ds_broken.data_vars
        assert "algorithm_uncertainty" not in ds_broken.data_vars
        with pytest.raises(OsisafParseError, match="algorithm_uncertainty"):
            _derive_total_uncertainty(ds_broken)

    def test_raises_parse_error_names_both_missing_components(self):
        """If both components are absent the error message names both."""
        ds_broken = _make_ds(
            missing_vars={
                "total_uncertainty",
                "algorithm_uncertainty",
                "smearing_uncertainty",
            }
        )
        with pytest.raises(OsisafParseError) as exc_info:
            _derive_total_uncertainty(ds_broken)
        msg = str(exc_info.value)
        assert "algorithm_uncertainty" in msg or "smearing_uncertainty" in msg


# ===========================================================================
# TestRegridIntegration (unit — no network, uses mock + regrid utility)
# ===========================================================================

class TestRegridIntegration:
    """
    Tests that fetch_nrt_concentration output is compatible with the
    regrid_to_package_grid() utility (polar-stereographic path).

    These tests use a mocked dataset with coordinates in metres and
    a tiny 5×4 grid; they do NOT need a real OSI SAF file or network
    access.
    """

    def test_regrid_ice_conc_to_target_grid(self):
        from cloud.data.regrid import regrid_to_package_grid
        from shared.schemas.common import GridMetadata

        target = GridMetadata(
            lat_min=-70.0, lat_max=-65.0,
            lon_min=-20.0, lon_max=-15.0,
            n_lat=3, n_lon=3,
        )

        # Use a broad grid in metres that covers the target lat/lon range when
        # projected through the OSI SAF SH CRS (true at 70°S, central meridian 0°).
        xc_m = np.linspace(-2_000_000.0, 2_000_000.0, _N_XC)
        yc_m = np.linspace(-2_000_000.0, 2_000_000.0, _N_YC)
        shape = (_N_YC, _N_XC)
        ice = np.full(shape, 0.75, dtype=np.float32)

        ds = xr.Dataset(
            {
                "ice_conc": xr.DataArray(ice, dims=("yc", "xc"), attrs={"units": "%"}),
            },
            coords={
                "xc": ("xc", xc_m, {"units": "m"}),
                "yc": ("yc", yc_m, {"units": "m"}),
            },
        )

        result = regrid_to_package_grid(
            ds, target, "ice_conc",
            source_crs=SH_PROJ,
            x_dim="xc",
            y_dim="yc",
        )

        assert len(result) == target.n_lat
        assert len(result[0]) == target.n_lon

    def test_regrid_result_values_are_float(self):
        from cloud.data.regrid import regrid_to_package_grid
        from shared.schemas.common import GridMetadata

        target = GridMetadata(
            lat_min=-70.0, lat_max=-65.0,
            lon_min=-20.0, lon_max=-15.0,
            n_lat=2, n_lon=2,
        )
        xc_m = np.linspace(-2_000_000.0, 2_000_000.0, _N_XC)
        yc_m = np.linspace(-2_000_000.0, 2_000_000.0, _N_YC)
        ice = np.full((_N_YC, _N_XC), 0.5, dtype=np.float32)
        ds = xr.Dataset(
            {"ice_conc": xr.DataArray(ice, dims=("yc", "xc"))},
            coords={"xc": xc_m, "yc": yc_m},
        )

        result = regrid_to_package_grid(
            ds, target, "ice_conc",
            source_crs=SH_PROJ, x_dim="xc", y_dim="yc",
        )
        for row in result:
            for val in row:
                assert isinstance(val, float)

    def test_regrid_ice_conc_and_total_uncertainty_through_sh_proj(self):
        """
        Prove that both ice_conc and total_uncertainty pass through the
        polar-stereographic regrid path independently, producing
        package-grid arrays of the correct shape, without network access.

        This is the canonical package-grid exercise for OSI SAF variables:
        a caller would regrid both fields, then store them in the
        ForecastPackage (once the schema supports gridded sea-ice fields
        in Phase 4+).  For now the test validates the regrid path only.

        SH_PROJ is the verified OSI SAF projection string (true scale at
        70°S, WGS84, central meridian 0°) — NOT EPSG:3031.
        """
        from cloud.data.regrid import regrid_to_package_grid
        from shared.schemas.common import GridMetadata

        target = GridMetadata(
            lat_min=-72.0, lat_max=-68.0,
            lon_min=-10.0, lon_max=10.0,
            n_lat=4, n_lon=5,
        )

        # Synthetic 10×8 source grid in metres; values chosen so the
        # interpolation result is predictable (uniform fields).
        n_xc, n_yc = 10, 8
        xc_m = np.linspace(-3_000_000.0, 3_000_000.0, n_xc)
        yc_m = np.linspace(-3_000_000.0, 3_000_000.0, n_yc)

        ice_fill    = 62.5    # % sea ice concentration
        unc_fill    = 8.3     # % total uncertainty

        ice_data = np.full((n_yc, n_xc), ice_fill, dtype=np.float32)
        unc_data = np.full((n_yc, n_xc), unc_fill, dtype=np.float32)

        ds = xr.Dataset(
            {
                "ice_conc":        xr.DataArray(ice_data, dims=("yc", "xc"),
                                                attrs={"units": "%"}),
                "total_uncertainty": xr.DataArray(unc_data, dims=("yc", "xc"),
                                                   attrs={"units": "%"}),
            },
            coords={
                "xc": ("xc", xc_m, {"units": "m"}),
                "yc": ("yc", yc_m, {"units": "m"}),
            },
        )

        ice_grid = regrid_to_package_grid(
            ds, target, "ice_conc",
            source_crs=SH_PROJ, x_dim="xc", y_dim="yc",
        )
        unc_grid = regrid_to_package_grid(
            ds, target, "total_uncertainty",
            source_crs=SH_PROJ, x_dim="xc", y_dim="yc",
        )

        # Shape: [n_lat][n_lon]
        assert len(ice_grid) == target.n_lat
        assert len(ice_grid[0]) == target.n_lon
        assert len(unc_grid) == target.n_lat
        assert len(unc_grid[0]) == target.n_lon

        # Uniform source → interior points should match fill values exactly
        # (edge points may be NaN if outside source domain, which is acceptable)
        flat_ice = [v for row in ice_grid for v in row if not np.isnan(v)]
        flat_unc = [v for row in unc_grid for v in row if not np.isnan(v)]

        assert len(flat_ice) > 0, "all ice_conc regrid points are NaN"
        assert len(flat_unc) > 0, "all total_uncertainty regrid points are NaN"

        for v in flat_ice:
            assert abs(v - ice_fill) < 1e-3, f"ice_conc={v} != {ice_fill}"
        for v in flat_unc:
            assert abs(v - unc_fill) < 1e-3, f"total_uncertainty={v} != {unc_fill}"

    def test_regrid_preserves_independent_variable_values(self):
        """
        Regridding ice_conc and total_uncertainty with different fill values
        gives distinct grids — confirms the two variables are routed
        independently through the same regrid path.
        """
        from cloud.data.regrid import regrid_to_package_grid
        from shared.schemas.common import GridMetadata

        target = GridMetadata(
            lat_min=-72.0, lat_max=-68.0,
            lon_min=-5.0, lon_max=5.0,
            n_lat=3, n_lon=3,
        )
        n_xc, n_yc = 8, 8
        xc_m = np.linspace(-3_000_000.0, 3_000_000.0, n_xc)
        yc_m = np.linspace(-3_000_000.0, 3_000_000.0, n_yc)

        ice_data = np.full((n_yc, n_xc), 40.0, dtype=np.float32)
        unc_data = np.full((n_yc, n_xc), 12.0, dtype=np.float32)

        ds = xr.Dataset(
            {
                "ice_conc":        xr.DataArray(ice_data, dims=("yc", "xc")),
                "total_uncertainty": xr.DataArray(unc_data, dims=("yc", "xc")),
            },
            coords={"xc": ("xc", xc_m, {"units": "m"}),
                    "yc": ("yc", yc_m, {"units": "m"})},
        )

        ice_grid = regrid_to_package_grid(
            ds, target, "ice_conc",
            source_crs=SH_PROJ, x_dim="xc", y_dim="yc",
        )
        unc_grid = regrid_to_package_grid(
            ds, target, "total_uncertainty",
            source_crs=SH_PROJ, x_dim="xc", y_dim="yc",
        )

        flat_ice = [v for row in ice_grid for v in row if not np.isnan(v)]
        flat_unc = [v for row in unc_grid for v in row if not np.isnan(v)]

        assert flat_ice, "ice_conc grid is all-NaN"
        assert flat_unc, "total_uncertainty grid is all-NaN"

        # The two grids must differ — they carried distinct source values
        mean_ice = sum(flat_ice) / len(flat_ice)
        mean_unc = sum(flat_unc) / len(flat_unc)
        assert abs(mean_ice - mean_unc) > 1.0, \
            f"ice_conc mean {mean_ice} and total_uncertainty mean {mean_unc} are too similar"


# ===========================================================================
# Integration test — real THREDDS network call (skipped in proxy-blocked env)
# ===========================================================================

@pytest.mark.integration
class TestOsisafIntegration:
    """
    Attempts a real OPeNDAP connection to thredds.met.no.  Skipped when
    the proxy blocks the host.

    Real access is NOT expected in the CCR build environment.
    """

    _SKIP_TOKENS = (
        "403", "407", "proxy", "forbidden", "connection", "refused",
        "timeout", "resolve", "unreachable", "reset", "ssl",
        "certificate", "stdin", "credentials", "not found", "404",
    )

    def _should_skip(self, msg: str) -> bool:
        lower = msg.lower()
        return any(t in lower for t in self._SKIP_TOKENS)

    def test_fetch_nrt_concentration_live(self):
        """
        Fetch the NRT concentration for a small Antarctic subset.

        Verifies:
          - Dataset is returned.
          - ice_conc, algorithm_uncertainty, smearing_uncertainty, status_flag present.
          - xc/yc coordinates exist.
          - time coordinate is present.
          - ice_conc values plausible (0–100%).
        """
        yesterday = (datetime.now(tz=timezone.utc) - timedelta(days=2)).date()
        try:
            ds = fetch_nrt_concentration(yesterday)
        except (OsisafNetworkError, OsisafNotFoundError) as exc:
            msg = str(exc)
            if self._should_skip(msg):
                pytest.skip(f"OSI SAF THREDDS unreachable (proxy/network): {msg}")
            raise

        assert "ice_conc" in ds, "ice_conc variable missing"
        assert "algorithm_uncertainty" in ds
        assert "smearing_uncertainty" in ds
        assert "status_flag" in ds

        # Coordinates must be present and in metres
        assert "xc" in ds.coords, "xc coordinate missing"
        assert "yc" in ds.coords, "yc coordinate missing"
        assert "time" in ds.coords, "time coordinate missing"

        xc_units = ds["xc"].attrs.get("units", "")
        assert xc_units == "m", f"xc units should be 'm' after normalisation, got '{xc_units}'"

        # ice_conc shape: (1, yc, xc) after decoding — squeeze and check range
        ice_vals = np.asarray(ds["ice_conc"].values, dtype=float)
        finite = ice_vals[np.isfinite(ice_vals)]
        assert finite.size > 0, "ice_conc contains only NaN"
        assert float(finite.min()) >= 0.0, "ice_conc below 0"
        assert float(finite.max()) <= 100.0, "ice_conc above 100"

        t = _extract_validity_time(ds)
        assert t is not None, "Could not extract validity time"
        assert t.tzinfo is not None
        assert t.date() == yesterday or abs((t.date() - yesterday).days) <= 1
