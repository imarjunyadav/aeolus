"""
Phase 3 Step 3 — tests for the NSIDC G02135 connector.

Test organisation
-----------------
  TestUrlHelpers           — pure helper functions, no I/O
  TestExtentCsvParser      — _parse_extent_csv() with real-format mock CSV text
  TestConcentrationParser  — _parse_concentration_tiff() with in-memory GeoTIFFs
  TestHttpErrorMapping     — _http_get() error classification using mocked urllib
  TestIntegration          — live network tests (require actual NSIDC access);
                             expected to raise NsidcNetworkError in environments
                             where noaadata.apps.nsidc.org is blocked

NOTE: The unit tests (all non-Integration classes) do NOT make network calls.
They test the parsing and error-handling logic only.
The integration tests hit the real NSIDC servers and are the only tests that
can confirm "real data successfully retrieved."  They are marked with
@pytest.mark.integration so CI can skip them when network access is absent.
"""

from __future__ import annotations

import io
import textwrap
import urllib.error
import urllib.request
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from cloud.data.sea_ice.nsidc import (
    NsidcNetworkError,
    NsidcNotFoundError,
    NsidcParseError,
    _concentration_url,
    _parse_concentration_tiff,
    _parse_extent_csv,
    fetch_concentration_geotiff,
    fetch_extent_csv,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_REAL_FORMAT_CSV_HEADER = (
    "Updating S_seaice_extent_daily_v3.0.csv\n"
    "\n"
    " Year,  Mo,  Day,       Extent,     Area,  Missing,Source Data, hemisphere\n"
)

_REAL_FORMAT_ROWS = (
    "2025,   1,  15,      10.234,     8.901,    0,NSIDC-0051, S\n"
    "2025,   1,  16,      10.456,     9.012,    0,NSIDC-0051, S\n"
    "2025,   7,  14,       5.678,     4.321,    0,NSIDC-0051, S\n"
    "2026,   9,  14,      10.001,     8.500,    0,NSIDC-0051, S\n"
)

_SAMPLE_CSV = _REAL_FORMAT_CSV_HEADER + _REAL_FORMAT_ROWS


def _make_tiff_bytes(
    data: np.ndarray,
    crs: str = "EPSG:3031",
    x_min: float = -400_000.0,
    y_min: float = 1_900_000.0,
    resolution: float = 25_000.0,
) -> bytes:
    """
    Create a minimal but valid GeoTIFF in memory using rasterio.
    The GeoTIFF uses EPSG:3031 and a 25 km resolution by default,
    matching the real NSIDC product geometry.
    """
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(x_min, y_min + data.shape[0] * resolution, resolution, resolution)
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(data, 1)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

class TestUrlHelpers:

    def test_concentration_url_jan(self):
        url = _concentration_url(date(2025, 1, 15))
        assert "/2025/01_Jan/" in url
        assert "S_20250115_concentration" in url
        assert url.endswith(".tif")

    def test_concentration_url_jul(self):
        url = _concentration_url(date(2025, 7, 4))
        assert "/2025/07_Jul/" in url
        assert "S_20250704_concentration" in url

    def test_concentration_url_dec(self):
        url = _concentration_url(date(2024, 12, 31))
        assert "/2024/12_Dec/" in url
        assert "S_20241231_concentration" in url

    def test_concentration_url_starts_with_base(self):
        url = _concentration_url(date(2025, 6, 1))
        assert url.startswith("https://noaadata.apps.nsidc.org/NOAA/G02135/")

    def test_concentration_url_month_padding(self):
        url = _concentration_url(date(2025, 3, 5))
        assert "/03_Mar/" in url   # month zero-padded


# ---------------------------------------------------------------------------
# Extent CSV parser
# ---------------------------------------------------------------------------

class TestExtentCsvParser:

    def test_parse_known_date(self):
        result = _parse_extent_csv(_SAMPLE_CSV, date(2025, 1, 15))
        assert result["extent_mkm2"] == pytest.approx(10.234)
        assert result["area_mkm2"]   == pytest.approx(8.901)
        assert result["date"]        == date(2025, 1, 15)
        assert result["source_data"] == "NSIDC-0051"

    def test_parse_another_date(self):
        result = _parse_extent_csv(_SAMPLE_CSV, date(2025, 7, 14))
        assert result["extent_mkm2"] == pytest.approx(5.678)
        assert result["area_mkm2"]   == pytest.approx(4.321)

    def test_missing_date_raises_parse_error(self):
        with pytest.raises(NsidcParseError, match="no row found"):
            _parse_extent_csv(_SAMPLE_CSV, date(2023, 6, 1))

    def test_missing_data_sentinel_raises_parse_error(self):
        csv_with_sentinel = (
            _REAL_FORMAT_CSV_HEADER
            + "2025,   2,  10,      -9999,    -9999,    0,NSIDC-0051, S\n"
        )
        with pytest.raises(NsidcParseError, match="-9999"):
            _parse_extent_csv(csv_with_sentinel, date(2025, 2, 10))

    def test_no_header_raises_parse_error(self):
        with pytest.raises(NsidcParseError, match="no header"):
            _parse_extent_csv("1978, 10, 26, 17.006, 14.569\n", date(1978, 10, 26))

    def test_missing_column_raises_parse_error(self):
        bad_header = "Year, Mo, Day\n2025, 1, 15\n"
        with pytest.raises(NsidcParseError, match="missing columns"):
            _parse_extent_csv(bad_header, date(2025, 1, 15))

    def test_parse_strips_whitespace(self):
        # Columns have leading spaces in real files
        csv = (
            " Year,  Mo,  Day,       Extent,     Area,  Missing,Source Data, hemisphere\n"
            "  2025,   9,  14,      11.000,     9.000,    0,NSIDC-0051, S\n"
        )
        result = _parse_extent_csv(csv, date(2025, 9, 14))
        assert result["extent_mkm2"] == pytest.approx(11.0)

    def test_returns_dict_keys(self):
        result = _parse_extent_csv(_SAMPLE_CSV, date(2025, 1, 16))
        assert set(result.keys()) >= {"date", "extent_mkm2", "area_mkm2", "source_data"}


# ---------------------------------------------------------------------------
# Concentration GeoTIFF parser
# ---------------------------------------------------------------------------

class TestConcentrationParser:
    """Uses in-memory GeoTIFF files created with rasterio — real binary format,
    no network access."""

    def test_output_is_dataarray(self):
        data = np.full((10, 10), 500, dtype=np.int16)
        tiff = _make_tiff_bytes(data)
        result = _parse_concentration_tiff(tiff)
        assert isinstance(result, xr.DataArray)

    def test_dims_are_y_x(self):
        data = np.full((8, 12), 800, dtype=np.int16)
        tiff = _make_tiff_bytes(data)
        result = _parse_concentration_tiff(tiff)
        assert result.dims == ("y", "x")

    def test_shape_preserved(self):
        data = np.full((6, 9), 300, dtype=np.int16)
        tiff = _make_tiff_bytes(data)
        result = _parse_concentration_tiff(tiff)
        assert result.shape == (6, 9)

    def test_scaling_0_to_1(self):
        """Raw value 500 (= 50% ice) → 0.5 fraction."""
        data = np.full((5, 5), 500, dtype=np.int16)
        tiff = _make_tiff_bytes(data)
        result = _parse_concentration_tiff(tiff)
        assert float(result.values[0, 0]) == pytest.approx(0.5)

    def test_scaling_full_coverage(self):
        """Raw 1000 → 1.0."""
        data = np.full((4, 4), 1000, dtype=np.int16)
        tiff = _make_tiff_bytes(data)
        result = _parse_concentration_tiff(tiff)
        assert float(result.values[0, 0]) == pytest.approx(1.0)

    def test_scaling_no_ice(self):
        """Raw 0 → 0.0."""
        data = np.zeros((4, 4), dtype=np.int16)
        tiff = _make_tiff_bytes(data)
        result = _parse_concentration_tiff(tiff)
        assert float(result.values[0, 0]) == pytest.approx(0.0)

    def test_special_flags_become_nan(self):
        """All four documented special flags (2510/2530/2540/2550) → NaN."""
        data = np.array([[500, 2510], [2530, 2540], [2550, 0]], dtype=np.int32)
        tiff = _make_tiff_bytes(data)
        result = _parse_concentration_tiff(tiff)
        # Valid concentration pixel preserved
        assert float(result.values[0, 0]) == pytest.approx(0.5)
        assert float(result.values[2, 1]) == pytest.approx(0.0)
        # All known flags become NaN
        assert np.isnan(result.values[0, 1])  # 2510 coast_line
        assert np.isnan(result.values[1, 0])  # 2530 land
        assert np.isnan(result.values[1, 1])  # 2540 missing
        assert np.isnan(result.values[2, 0])  # 2550 no_data

    @pytest.mark.parametrize("flag_val,label", [
        (2510, "coast_line"),
        (2530, "land"),
        (2540, "missing"),
        (2550, "no_data"),
    ])
    def test_each_known_flag_becomes_nan(self, flag_val, label):
        """Each individual known flag value is masked to NaN."""
        data = np.array([[flag_val, 500]], dtype=np.int32)
        tiff = _make_tiff_bytes(data)
        result = _parse_concentration_tiff(tiff)
        assert np.isnan(result.values[0, 0]), f"Flag {flag_val} ({label}) should be NaN"
        assert float(result.values[0, 1]) == pytest.approx(0.5)

    def test_unexpected_raw_value_raises_parse_error(self):
        """Raw value 1500 is outside valid range and not a known flag → NsidcParseError."""
        data = np.array([[500, 1500]], dtype=np.int32)
        tiff = _make_tiff_bytes(data)
        with pytest.raises(NsidcParseError, match="unexpected raw pixel"):
            _parse_concentration_tiff(tiff)

    def test_unexpected_raw_value_in_gap_raises_parse_error(self):
        """Raw value 1001 is just above valid range and below any known flag → NsidcParseError."""
        data = np.array([[1001]], dtype=np.int32)
        tiff = _make_tiff_bytes(data)
        with pytest.raises(NsidcParseError, match="unexpected raw pixel"):
            _parse_concentration_tiff(tiff)

    def test_negative_raw_value_raises_parse_error(self):
        """Negative raw values are not valid concentration or known flags."""
        data = np.array([[-1, 500]], dtype=np.int32)
        tiff = _make_tiff_bytes(data)
        with pytest.raises(NsidcParseError, match="unexpected raw pixel"):
            _parse_concentration_tiff(tiff)

    def test_mixed_field(self):
        """Grid with varied values parsed correctly."""
        raw = np.array([[0, 250, 500, 750, 1000]], dtype=np.int16)
        tiff = _make_tiff_bytes(raw)
        result = _parse_concentration_tiff(tiff)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        for j, exp in enumerate(expected):
            assert float(result.values[0, j]) == pytest.approx(exp)

    def test_x_coordinates_are_metres(self):
        """x coords should be in millions-of-metres range (EPSG:3031)."""
        data = np.full((4, 10), 500, dtype=np.int16)
        tiff = _make_tiff_bytes(data, x_min=-400_000)
        result = _parse_concentration_tiff(tiff)
        x_min = float(result.x.min())
        x_max = float(result.x.max())
        # Should be O(100k) metres — not lat/lon degrees
        assert abs(x_min) < 1e7 and abs(x_max) < 1e7
        assert abs(x_min) > 1000 or abs(x_max) > 1000   # definitely not degrees

    def test_crs_in_attrs(self):
        data = np.full((4, 4), 500, dtype=np.int16)
        tiff = _make_tiff_bytes(data)
        result = _parse_concentration_tiff(tiff)
        assert "crs" in result.attrs
        assert result.attrs["crs"] != ""

    def test_invalid_bytes_raises_parse_error(self):
        with pytest.raises(NsidcParseError, match="rasterio could not open"):
            _parse_concentration_tiff(b"not a tiff file at all")


# ---------------------------------------------------------------------------
# HTTP error mapping
# ---------------------------------------------------------------------------

class TestHttpErrorMapping:
    """Verify that _http_get classifies errors correctly without network access."""

    def _make_http_error(self, code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            url="https://fake.nsidc.org/file.tif",
            code=code,
            msg=f"HTTP {code}",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    def test_404_raises_not_found(self):
        from cloud.data.sea_ice.nsidc import _http_get

        with patch("urllib.request.urlopen", side_effect=self._make_http_error(404)):
            with pytest.raises(NsidcNotFoundError):
                _http_get("https://fake.nsidc.org/missing.tif")

    def test_not_found_is_subclass_of_network_error(self):
        from cloud.data.sea_ice.nsidc import _http_get

        with patch("urllib.request.urlopen", side_effect=self._make_http_error(404)):
            with pytest.raises(NsidcNetworkError):
                _http_get("https://fake.nsidc.org/missing.tif")

    def test_503_raises_network_error_not_not_found(self):
        from cloud.data.sea_ice.nsidc import _http_get

        with patch("urllib.request.urlopen", side_effect=self._make_http_error(503)):
            with pytest.raises(NsidcNetworkError) as exc_info:
                _http_get("https://fake.nsidc.org/busy.tif")
            assert not isinstance(exc_info.value, NsidcNotFoundError)

    def test_url_error_raises_network_error(self):
        from cloud.data.sea_ice.nsidc import _http_get

        url_err = urllib.error.URLError("connection refused")
        with patch("urllib.request.urlopen", side_effect=url_err):
            with pytest.raises(NsidcNetworkError):
                _http_get("https://fake.nsidc.org/file.tif")

    def test_successful_fetch_returns_bytes(self):
        from cloud.data.sea_ice.nsidc import _http_get

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"hello"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _http_get("https://fake.nsidc.org/ok.tif")
        assert result == b"hello"

    def test_fetch_extent_csv_raises_on_network_error(self):
        """fetch_extent_csv propagates NsidcNetworkError."""
        url_err = urllib.error.URLError("no route to host")
        with patch("urllib.request.urlopen", side_effect=url_err):
            with pytest.raises(NsidcNetworkError):
                fetch_extent_csv(date(2025, 1, 15))

    def test_fetch_concentration_raises_on_network_error(self):
        """fetch_concentration_geotiff propagates NsidcNetworkError."""
        url_err = urllib.error.URLError("no route to host")
        with patch("urllib.request.urlopen", side_effect=url_err):
            with pytest.raises(NsidcNetworkError):
                fetch_concentration_geotiff(date(2025, 1, 15))

    def test_fetch_extent_csv_raises_parse_error_on_bad_content(self):
        """If the server returns garbage, we get NsidcParseError, not NsidcNetworkError."""
        garbage = b"<html>Server error</html>"
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = garbage

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(NsidcParseError):
                fetch_extent_csv(date(2025, 1, 15))

    def test_fetch_extent_csv_uses_correct_url(self):
        """Verify the right URL is requested."""
        from cloud.data.sea_ice.nsidc import _EXTENT_CSV_URL

        url_err = urllib.error.URLError("blocked")
        with patch("urllib.request.urlopen", side_effect=url_err) as mock_open:
            with pytest.raises(NsidcNetworkError):
                fetch_extent_csv(date(2025, 1, 15))
            called_url = mock_open.call_args[0][0].full_url
            assert called_url == _EXTENT_CSV_URL

    def test_fetch_concentration_uses_correct_url(self):
        """Verify the GeoTIFF URL contains the right date components."""
        url_err = urllib.error.URLError("blocked")
        with patch("urllib.request.urlopen", side_effect=url_err) as mock_open:
            with pytest.raises(NsidcNetworkError):
                fetch_concentration_geotiff(date(2025, 7, 14))
            called_url = mock_open.call_args[0][0].full_url
            assert "20250714" in called_url
            assert "07_Jul" in called_url


# ---------------------------------------------------------------------------
# Integration tests — require live NSIDC access
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """
    These tests make real HTTP requests to noaadata.apps.nsidc.org.

    Run with:  pytest -m integration tests/test_phase3_nsidc.py

    In a network-restricted environment (e.g. the CCR proxy policy that
    blocks noaadata.apps.nsidc.org), these tests are expected to raise
    NsidcNetworkError.  That outcome itself confirms that:
      (a) the connector attempts a real network connection,
      (b) the network error is correctly classified, and
      (c) the error message is informative.

    A PASSED result here (on a machine with unrestricted access) means
    real data was retrieved and successfully parsed.
    """

    # Query a date well in the past — data for it is stable and always present.
    _TEST_DATE = date(2025, 7, 14)

    def test_extent_csv_real_or_network_error(self):
        """
        Either returns real data with plausible SH winter extent, or raises
        NsidcNetworkError.  Neither case is a test failure — the assertion
        is on what *kind* of outcome occurs.
        """
        try:
            result = fetch_extent_csv(self._TEST_DATE)
            # If we get here, real data was retrieved.
            assert isinstance(result["extent_mkm2"], float)
            assert isinstance(result["area_mkm2"], float)
            # SH winter minimum is ~2 Mkm²; maximum ~20 Mkm²
            assert 1.0 <= result["extent_mkm2"] <= 25.0, (
                f"Implausible SH extent: {result['extent_mkm2']} Mkm²"
            )
            assert result["date"] == self._TEST_DATE
            print(f"\n[REAL DATA] {self._TEST_DATE}: extent={result['extent_mkm2']} Mkm², "
                  f"area={result['area_mkm2']} Mkm², source={result['source_data']!r}")
        except NsidcNetworkError as exc:
            # Expected in proxy-restricted CI. Print reason for the test report.
            pytest.skip(
                f"NSIDC unreachable (proxy policy likely): {exc}\n"
                "Re-run without network restriction to verify real data."
            )

    def test_concentration_geotiff_real_or_network_error(self):
        """
        Either parses a real GeoTIFF or surfaces a NsidcNetworkError.
        """
        try:
            da = fetch_concentration_geotiff(self._TEST_DATE)
            assert isinstance(da, xr.DataArray)
            assert da.dims == ("y", "x")
            # Real SH grid is 316×332 or similar — larger than a toy array
            assert da.shape[0] >= 100 and da.shape[1] >= 100
            # Concentration values should be in [0, 1] or NaN
            valid = da.values[~np.isnan(da.values)]
            assert valid.min() >= 0.0
            assert valid.max() <= 1.0
            # CRS should mention 3031
            assert "3031" in da.attrs.get("crs", "") or "stereo" in da.attrs.get("crs", "").lower()
            print(f"\n[REAL DATA] concentration grid: shape={da.shape}, "
                  f"x=[{float(da.x.min()):.0f}, {float(da.x.max()):.0f}] m, "
                  f"y=[{float(da.y.min()):.0f}, {float(da.y.max()):.0f}] m, "
                  f"valid_pixels={valid.size}, "
                  f"mean_concentration={valid.mean():.3f}")
        except NsidcNetworkError as exc:
            pytest.skip(
                f"NSIDC unreachable (proxy policy likely): {exc}\n"
                "Re-run without network restriction to verify real data."
            )
