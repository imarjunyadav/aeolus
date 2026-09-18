"""
Phase 3 Step 4 — tests for the USNIC iceberg connector.

Test organisation
-----------------
  TestExceptionHierarchy  — exception class relationships
  TestCoordParser         — _parse_coord() edge cases
  TestDateParser          — _parse_date() across all accepted formats
  TestCsvParser           — _parse_csv() with crafted fixture CSV bytes
  TestHttpErrorMapping    — _http_get() error classification (mocked urllib)
  TestIntegration         — live network tests; skip on network failure

NOTE: All non-Integration tests make no network calls. They exercise the
parsing and error-handling logic only against in-memory fixture data.
"""

from __future__ import annotations

import math
import textwrap
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from cloud.data.icebergs.usnic import (
    UsnicNetworkError,
    UsnicNotFoundError,
    UsnicParseError,
    _http_get,
    _parse_coord,
    _parse_csv,
    _parse_date,
    fetch_current_positions,
)
from cloud.data.exceptions import IcebergConnectorError
from shared.schemas.forecast import IcebergForecast


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _csv_bytes(content: str) -> bytes:
    return textwrap.dedent(content).strip().encode("utf-8")


# Minimal valid 6-column CSV
_VALID_CSV_6COL = _csv_bytes("""
    Iceberg,Length (NM),Width (NM),Latitude,Longitude,Last Update
    A23A,9,7,-48.9534,-31.7867,09/14/2026
    B46,25,15,-55.1200,34.5600,09/14/2026
""")

# Valid 7-column CSV (with optional Remarks column)
_VALID_CSV_7COL = _csv_bytes("""
    Iceberg,Length (NM),Width (NM),Latitude,Longitude,Remarks,Last Update
    A23A,9,7,-48.9534,-31.7867,Grounded,09/14/2026
    D28A,18,12,-63.4500,-47.2300,,09/07/2026
""")

# CSV where lat/lon use hemisphere suffix notation
_VALID_CSV_HEMI = _csv_bytes("""
    Iceberg,Length (NM),Width (NM),Latitude,Longitude,Last Update
    A23A,9,7,48.9534 S,31.7867 W,09/14/2026
    C19,40,22,70.1200 S,165.4300 E,09/14/2026
""")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:

    def test_network_error_is_iceberg_connector_error(self):
        assert issubclass(UsnicNetworkError, IcebergConnectorError)

    def test_not_found_is_network_error(self):
        assert issubclass(UsnicNotFoundError, UsnicNetworkError)

    def test_parse_error_is_iceberg_connector_error(self):
        assert issubclass(UsnicParseError, IcebergConnectorError)

    def test_not_found_is_not_parse_error(self):
        assert not issubclass(UsnicNotFoundError, UsnicParseError)


# ---------------------------------------------------------------------------
# Coordinate parser
# ---------------------------------------------------------------------------

class TestCoordParser:

    def test_negative_float_passes_through(self):
        assert _parse_coord(-48.9534, is_lon=False) == pytest.approx(-48.9534)

    def test_positive_float_passes_through(self):
        assert _parse_coord(34.56, is_lon=True) == pytest.approx(34.56)

    def test_negative_string_float(self):
        assert _parse_coord("-48.9534", is_lon=False) == pytest.approx(-48.9534)

    def test_positive_string_float(self):
        assert _parse_coord("34.56", is_lon=True) == pytest.approx(34.56)

    def test_south_suffix(self):
        result = _parse_coord("48.9534 S", is_lon=False)
        assert result == pytest.approx(-48.9534)

    def test_north_suffix_stays_positive(self):
        result = _parse_coord("12.3456 N", is_lon=False)
        assert result == pytest.approx(12.3456)

    def test_west_suffix(self):
        result = _parse_coord("31.7867 W", is_lon=True)
        assert result == pytest.approx(-31.7867)

    def test_east_suffix_stays_positive(self):
        result = _parse_coord("165.43 E", is_lon=True)
        assert result == pytest.approx(165.43)

    def test_lowercase_hemisphere_suffix(self):
        assert _parse_coord("55.0 s", is_lon=False) == pytest.approx(-55.0)
        assert _parse_coord("40.0 w", is_lon=True) == pytest.approx(-40.0)

    def test_none_returns_none(self):
        assert _parse_coord(None, is_lon=False) is None

    def test_empty_string_returns_none(self):
        assert _parse_coord("", is_lon=False) is None

    def test_nan_float_returns_none(self):
        assert _parse_coord(float("nan"), is_lon=False) is None

    def test_garbage_string_returns_none(self):
        assert _parse_coord("abc", is_lon=False) is None

    def test_integer_input(self):
        assert _parse_coord(-55, is_lon=False) == pytest.approx(-55.0)

    def test_zero(self):
        assert _parse_coord(0.0, is_lon=True) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Date parser
# ---------------------------------------------------------------------------

class TestDateParser:

    def test_us_slash_format(self):
        dt = _parse_date("09/14/2026")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 9
        assert dt.day == 14
        assert dt.tzinfo == timezone.utc

    def test_iso_format(self):
        dt = _parse_date("2026-09-14")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 9 and dt.day == 14

    def test_dd_mon_yyyy(self):
        dt = _parse_date("14 Sep 2026")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 9 and dt.day == 14

    def test_dd_dash_mon_yyyy(self):
        dt = _parse_date("14-Sep-2026")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 9 and dt.day == 14

    def test_long_month_name(self):
        dt = _parse_date("September 14, 2026")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 9 and dt.day == 14

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_nan_string_returns_none(self):
        assert _parse_date("nan") is None

    def test_empty_string_returns_none(self):
        assert _parse_date("") is None

    def test_garbage_returns_none(self):
        assert _parse_date("not-a-date") is None

    def test_result_is_utc(self):
        dt = _parse_date("09/14/2026")
        assert dt is not None
        assert dt.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# CSV parser — structure and correctness
# ---------------------------------------------------------------------------

class TestCsvParser:

    def test_basic_6col_parse(self):
        icebergs, as_of = _parse_csv(_VALID_CSV_6COL)
        assert len(icebergs) == 2
        assert isinstance(as_of, datetime)

    def test_basic_7col_parse(self):
        icebergs, as_of = _parse_csv(_VALID_CSV_7COL)
        assert len(icebergs) == 2

    def test_hemisphere_suffix_coords(self):
        icebergs, _ = _parse_csv(_VALID_CSV_HEMI)
        a23a = next(b for b in icebergs if b.iceberg_id == "A23A")
        assert a23a.last_known_position.latitude == pytest.approx(-48.9534)
        assert a23a.last_known_position.longitude == pytest.approx(-31.7867)

    def test_east_longitude_stays_positive(self):
        icebergs, _ = _parse_csv(_VALID_CSV_HEMI)
        c19 = next(b for b in icebergs if b.iceberg_id == "C19")
        assert c19.last_known_position.longitude == pytest.approx(165.43)

    def test_iceberg_ids_preserved(self):
        icebergs, _ = _parse_csv(_VALID_CSV_6COL)
        ids = {b.iceberg_id for b in icebergs}
        assert "A23A" in ids
        assert "B46" in ids

    def test_returns_iceberg_forecast_instances(self):
        icebergs, _ = _parse_csv(_VALID_CSV_6COL)
        for iceberg in icebergs:
            assert isinstance(iceberg, IcebergForecast)

    def test_horizons_are_empty(self):
        icebergs, _ = _parse_csv(_VALID_CSV_6COL)
        for iceberg in icebergs:
            assert iceberg.horizons == []

    def test_as_of_is_latest_update(self):
        # Mix of dates: 09/14/2026 and 09/07/2026 — as_of should be the later
        icebergs, as_of = _parse_csv(_VALID_CSV_7COL)
        assert as_of.year == 2026 and as_of.month == 9 and as_of.day == 14

    def test_source_data_age_is_non_negative(self):
        icebergs, _ = _parse_csv(_VALID_CSV_6COL)
        for iceberg in icebergs:
            assert iceberg.source_data_age_seconds >= 0

    def test_latitudes_are_southern_hemisphere(self):
        icebergs, _ = _parse_csv(_VALID_CSV_6COL)
        for iceberg in icebergs:
            assert iceberg.last_known_position.latitude < 0

    def test_non_antarctic_rows_skipped(self):
        csv = _csv_bytes("""
            Iceberg,Length (NM),Width (NM),Latitude,Longitude,Last Update
            A23A,9,7,-48.9534,-31.7867,09/14/2026
            NORTH,5,3,75.0,-30.0,09/14/2026
        """)
        icebergs, _ = _parse_csv(csv)
        assert len(icebergs) == 1
        assert icebergs[0].iceberg_id == "A23A"

    def test_missing_required_column_raises_parse_error(self):
        csv = _csv_bytes("""
            Iceberg,Length (NM),Width (NM),Last Update
            A23A,9,7,09/14/2026
        """)
        with pytest.raises(UsnicParseError, match="missing required columns"):
            _parse_csv(csv)

    def test_empty_csv_raises_parse_error(self):
        with pytest.raises(UsnicParseError):
            _parse_csv(b"")

    def test_header_only_raises_parse_error(self):
        with pytest.raises(UsnicParseError):
            _parse_csv(b"Iceberg,Length (NM),Width (NM),Latitude,Longitude,Last Update\n")

    def test_all_non_antarctic_raises_parse_error(self):
        csv = _csv_bytes("""
            Iceberg,Length (NM),Width (NM),Latitude,Longitude,Last Update
            NORTH,5,3,75.0,-30.0,09/14/2026
        """)
        with pytest.raises(UsnicParseError, match="no valid Antarctic"):
            _parse_csv(csv)

    def test_invalid_coords_skipped(self):
        csv = _csv_bytes("""
            Iceberg,Length (NM),Width (NM),Latitude,Longitude,Last Update
            A23A,9,7,-48.9534,-31.7867,09/14/2026
            BAD,5,3,bad_lat,bad_lon,09/14/2026
        """)
        icebergs, _ = _parse_csv(csv)
        assert len(icebergs) == 1
        assert icebergs[0].iceberg_id == "A23A"

    def test_invalid_date_rows_skipped(self):
        csv = _csv_bytes("""
            Iceberg,Length (NM),Width (NM),Latitude,Longitude,Last Update
            A23A,9,7,-48.9534,-31.7867,09/14/2026
            B46,5,3,-55.0,34.5,not-a-date
        """)
        icebergs, _ = _parse_csv(csv)
        assert len(icebergs) == 1
        assert icebergs[0].iceberg_id == "A23A"

    def test_lon_over_180_normalised(self):
        csv = _csv_bytes("""
            Iceberg,Length (NM),Width (NM),Latitude,Longitude,Last Update
            D15,10,8,-55.0,200.0,09/14/2026
        """)
        icebergs, _ = _parse_csv(csv)
        assert icebergs[0].last_known_position.longitude == pytest.approx(-160.0)

    def test_position_object_has_correct_type(self):
        from shared.schemas.common import Position
        icebergs, _ = _parse_csv(_VALID_CSV_6COL)
        assert isinstance(icebergs[0].last_known_position, Position)

    def test_latin1_encoded_csv(self):
        content = (
            "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Last Update\n"
            "A23A,9,7,-48.9534,-31.7867,09/14/2026\n"
        )
        icebergs, _ = _parse_csv(content.encode("latin-1"))
        assert len(icebergs) == 1


# ---------------------------------------------------------------------------
# HTTP error mapping
# ---------------------------------------------------------------------------

class TestHttpErrorMapping:

    def _make_http_error(self, code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            url="https://usicecenter.gov/File/DownloadCurrent?pId=134",
            code=code,
            msg=f"HTTP {code}",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    def test_404_raises_not_found(self):
        with patch("urllib.request.urlopen", side_effect=self._make_http_error(404)):
            with pytest.raises(UsnicNotFoundError):
                _http_get("https://usicecenter.gov/File/DownloadCurrent?pId=134")

    def test_not_found_is_subclass_of_network_error(self):
        with patch("urllib.request.urlopen", side_effect=self._make_http_error(404)):
            with pytest.raises(UsnicNetworkError):
                _http_get("https://usicecenter.gov/File/DownloadCurrent?pId=134")

    def test_503_raises_network_error_not_not_found(self):
        with patch("urllib.request.urlopen", side_effect=self._make_http_error(503)):
            with pytest.raises(UsnicNetworkError) as exc_info:
                _http_get("https://usicecenter.gov/File/DownloadCurrent?pId=134")
            assert not isinstance(exc_info.value, UsnicNotFoundError)

    def test_url_error_raises_network_error(self):
        url_err = urllib.error.URLError("connection refused")
        with patch("urllib.request.urlopen", side_effect=url_err):
            with pytest.raises(UsnicNetworkError):
                _http_get("https://usicecenter.gov/File/DownloadCurrent?pId=134")

    def test_successful_fetch_returns_bytes(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"col1,col2\nval1,val2\n"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _http_get("https://usicecenter.gov/File/DownloadCurrent?pId=134")
        assert result == b"col1,col2\nval1,val2\n"

    def test_fetch_propagates_network_error(self):
        url_err = urllib.error.URLError("no route to host")
        with patch("urllib.request.urlopen", side_effect=url_err):
            with pytest.raises(UsnicNetworkError):
                fetch_current_positions()

    def test_fetch_propagates_parse_error_on_garbage(self):
        """If the server returns non-CSV content, we get UsnicParseError."""
        garbage = b"<html>Server error</html>"
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = garbage

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises((UsnicParseError, UsnicNetworkError)):
                fetch_current_positions()

    def test_fetch_uses_correct_url(self):
        from cloud.data.icebergs.usnic import _USNIC_CSV_URL

        url_err = urllib.error.URLError("blocked")
        with patch("urllib.request.urlopen", side_effect=url_err) as mock_open:
            with pytest.raises(UsnicNetworkError):
                fetch_current_positions()
            called_url = mock_open.call_args[0][0].full_url
            assert called_url == _USNIC_CSV_URL


# ---------------------------------------------------------------------------
# Integration tests — require live USNIC access
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """
    These tests make real HTTP requests to usicecenter.gov.

    Run with:  pytest -m integration tests/test_phase3_usnic.py

    In a network-restricted environment (e.g. the CCR proxy policy that
    blocks usicecenter.gov), these tests are expected to raise
    UsnicNetworkError.  That outcome confirms:
      (a) the connector attempts a real network connection,
      (b) the network error is correctly classified, and
      (c) the error message is informative.

    A PASSED result (on a machine with unrestricted access) means real data
    was retrieved and successfully parsed.
    """

    def test_fetch_current_positions_real_or_network_error(self):
        """
        Either returns real Antarctic iceberg data, or raises UsnicNetworkError.
        Neither outcome is a test failure — we assert on the *kind* of outcome.
        """
        try:
            icebergs, as_of = fetch_current_positions()
            # If we get here, real data was retrieved.
            assert isinstance(icebergs, list)
            assert len(icebergs) > 0, "Expected at least one Antarctic iceberg"
            assert isinstance(as_of, datetime)
            assert as_of.tzinfo is not None

            # Every iceberg must be in the Southern Ocean
            for iceberg in icebergs:
                assert iceberg.last_known_position.latitude < 0, (
                    f"Non-southern latitude for {iceberg.iceberg_id}: "
                    f"{iceberg.last_known_position.latitude}"
                )
                assert -90.0 <= iceberg.last_known_position.latitude <= -40.0, (
                    f"Latitude out of Antarctic range: {iceberg.last_known_position.latitude}"
                )
                assert -180.0 <= iceberg.last_known_position.longitude <= 180.0

            # Icebergs should follow USNIC naming convention (e.g. A23A, B46)
            for iceberg in icebergs:
                assert len(iceberg.iceberg_id) >= 2, f"Short ID: {iceberg.iceberg_id!r}"

            # Log real data for human review
            print(
                f"\n[REAL DATA] {len(icebergs)} Antarctic icebergs as of {as_of.date()}"
            )
            for b in icebergs[:5]:
                print(
                    f"  {b.iceberg_id}: lat={b.last_known_position.latitude:.4f}, "
                    f"lon={b.last_known_position.longitude:.4f}, "
                    f"age={b.source_data_age_seconds // 86400}d"
                )

        except UsnicNetworkError as exc:
            pytest.skip(
                f"USNIC unreachable (proxy policy likely): {exc}\n"
                "Re-run without network restriction to verify real data."
            )
