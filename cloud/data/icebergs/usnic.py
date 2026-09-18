"""
USNIC iceberg position connector (operational current positions).

Product : Antarctic Iceberg Position CSV — weekly, all named icebergs ≥ 50 km².
Source  : https://usicecenter.gov/File/DownloadCurrent?pId=134
Auth    : None required.
Format  : CSV — Iceberg, Length (NM), Width (NM), Latitude, Longitude,
          [Remarks,] Last Update
          (7-column variant with Remarks is also observed in the wild)

Failure taxonomy
----------------
All public functions raise a subclass of IcebergConnectorError:

  UsnicNotFoundError(UsnicNetworkError)
      HTTP 404 — URL structure has changed or resource missing.

  UsnicNetworkError(IcebergConnectorError)
      Any other network or HTTP failure: connection refused, proxy denial,
      timeout, non-200 status code.

  UsnicParseError(IcebergConnectorError)
      The file was fetched but could not be parsed: unexpected structure,
      missing required columns, unparseable coordinates or dates, no
      Southern Ocean rows after filtering.

Coordinate convention
---------------------
The CSV Latitude and Longitude columns are signed decimal degrees.
Southern latitudes are negative (all Antarctic icebergs); western
longitudes are negative.  No N/S/E/W suffix is present in the machine-
readable CSV (the PDF report uses DMS+suffix for display, but that is a
different download product).

As a conservative fallback the parser also accepts values with a trailing
hemisphere letter, e.g. "48.9500 S" or "31.7867 W", in case an older
archive file uses that form.

All rows with lat >= -40° are silently skipped as non-Antarctic.

Last Update date
----------------
The "Last Update" field uses MM/DD/YYYY (US format, confirmed from the
USNIC HTML table and PDF companion report).  Additional formats are also
accepted for robustness against historical archive files.  All dates are
interpreted as UTC midnight.

Network access note
-------------------
This connector makes real outbound requests to usicecenter.gov.
In environments where that host is blocked (e.g. proxy policy 403), all
public functions raise UsnicNetworkError.  Real-data verification was not
possible in the CCR build environment; integration tests skip with the
proxy error message when the host is unreachable.
"""

from __future__ import annotations

import io
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from cloud.data.exceptions import IcebergConnectorError
from shared.schemas.common import Position
from shared.schemas.forecast import IcebergForecast


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class UsnicNetworkError(IcebergConnectorError):
    """Network or HTTP failure reaching USNIC (non-404)."""


class UsnicNotFoundError(UsnicNetworkError):
    """HTTP 404 — URL structure changed or resource missing."""


class UsnicParseError(IcebergConnectorError):
    """Downloaded CSV could not be parsed."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USNIC_CSV_URL = "https://usicecenter.gov/File/DownloadCurrent?pId=134"
_HTTP_TIMEOUT_S = 30
_USER_AGENT = "aeolus-usnic-connector/1.0"

# Icebergs north of this latitude are not Antarctic; skip them.
_SOUTHERN_OCEAN_LAT_MAX = -40.0

# Columns the connector recognises (lowercase, stripped).
# "Remarks" is optional — some versions of the file omit it.
_COL_ICEBERG    = "iceberg"
_COL_LENGTH_NM  = "length (nm)"
_COL_WIDTH_NM   = "width (nm)"
_COL_LATITUDE   = "latitude"
_COL_LONGITUDE  = "longitude"
_COL_LAST_UPDATE = "last update"

_REQUIRED_COLS = {_COL_ICEBERG, _COL_LATITUDE, _COL_LONGITUDE, _COL_LAST_UPDATE}

# Known USNIC date formats (tried in order).
_DATE_FORMATS = [
    "%m/%d/%Y",   # 09/14/2026  — most common US government format
    "%Y-%m-%d",   # 2026-09-14  — ISO
    "%d %b %Y",   # 14 Sep 2026
    "%d-%b-%Y",   # 14-Sep-2026
    "%B %d, %Y",  # September 14, 2026
]

# Hemisphere suffix pattern: optional whitespace + N/S/E/W
_HEMI_RE = re.compile(r"^\s*([\d.]+)\s*([NSEWnsew])\s*$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _http_get(url: str) -> bytes:
    """
    Fetch URL bytes.

    Raises
    ------
    UsnicNotFoundError  : HTTP 404
    UsnicNetworkError   : any other HTTP or connection failure
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UsnicNotFoundError(
                f"USNIC 404 — {url!r}: resource not found or URL structure changed"
            ) from exc
        raise UsnicNetworkError(
            f"USNIC HTTP {exc.code} fetching {url!r}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise UsnicNetworkError(
            f"USNIC network error fetching {url!r}: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise UsnicNetworkError(
            f"USNIC I/O error fetching {url!r}: {exc}"
        ) from exc


def _parse_coord(raw: Any, is_lon: bool) -> float | None:
    """
    Parse a coordinate value from the CSV.

    Accepts:
      - float or int already (pandas may parse as numeric)
      - string decimal: "-48.9500" or "48.9500"
      - string with hemisphere suffix: "48.9500 S", "31.7867 W"

    Returns signed decimal degrees, or None if unparseable.
    Southern/western hemisphere values are returned as negative floats.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        import math
        if math.isnan(raw):
            return None
        return float(raw)

    s = str(raw).strip()
    if not s:
        return None

    # Try plain float first
    try:
        return float(s)
    except ValueError:
        pass

    # Try hemisphere suffix: "48.9500 S"
    m = _HEMI_RE.match(s)
    if m:
        mag = float(m.group(1))
        hemi = m.group(2).upper()
        if hemi in ("S", "W"):
            return -mag
        return mag  # N or E

    return None


def _parse_date(raw: Any) -> datetime | None:
    """
    Parse a "Last Update" value from the CSV into a UTC midnight datetime.

    Returns None if the value is NaN/None or all formats fail.
    """
    if raw is None:
        return None
    if isinstance(raw, float):
        import math
        if math.isnan(raw):
            return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None

    # pandas may have already parsed it as a Timestamp
    if isinstance(raw, pd.Timestamp):
        return raw.to_pydatetime().replace(tzinfo=timezone.utc,
                                           hour=0, minute=0, second=0, microsecond=0)

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_csv(raw_bytes: bytes) -> tuple[list[IcebergForecast], datetime]:
    """
    Parse the raw USNIC CSV bytes into a list of IcebergForecast objects.

    Returns (icebergs, as_of_timestamp) where as_of_timestamp is the most
    recent "Last Update" date across all returned rows.

    Raises
    ------
    UsnicParseError : CSV has unexpected structure, missing columns, no
                      decodeable content, or zero valid Antarctic rows.
    """
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise UsnicParseError(f"USNIC CSV is not UTF-8 or Latin-1: {exc}") from exc

    try:
        df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    except Exception as exc:
        raise UsnicParseError(f"USNIC CSV could not be parsed by pandas: {exc}") from exc

    if df.empty:
        raise UsnicParseError("USNIC CSV: file is empty or has no data rows")

    # Normalise column names: lowercase and strip whitespace.
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise UsnicParseError(
            f"USNIC CSV: missing required columns {missing}; "
            f"found {list(df.columns)}"
        )

    now = datetime.now(tz=timezone.utc)
    icebergs: list[IcebergForecast] = []
    skipped_coord = 0
    skipped_date = 0
    skipped_domain = 0
    latest_timestamp: datetime | None = None

    for _, row in df.iterrows():
        iceberg_id = str(row[_COL_ICEBERG]).strip()
        if not iceberg_id or iceberg_id.lower() in ("nan", ""):
            continue

        # Parse coordinates
        lat = _parse_coord(row[_COL_LATITUDE], is_lon=False)
        lon = _parse_coord(row[_COL_LONGITUDE], is_lon=True)
        if lat is None or lon is None:
            skipped_coord += 1
            continue

        # Reject non-Antarctic rows
        if lat >= _SOUTHERN_OCEAN_LAT_MAX:
            skipped_domain += 1
            continue

        # Normalise longitude to [-180, 180]
        if lon > 180.0:
            lon -= 360.0
        elif lon < -180.0:
            lon += 360.0

        # Parse date
        last_update = _parse_date(row[_COL_LAST_UPDATE])
        if last_update is None:
            skipped_date += 1
            continue

        if latest_timestamp is None or last_update > latest_timestamp:
            latest_timestamp = last_update

        age_s = max(0, int((now - last_update).total_seconds()))

        icebergs.append(
            IcebergForecast(
                iceberg_id=iceberg_id,
                last_known_position=Position(latitude=lat, longitude=lon),
                last_known_timestamp=last_update,
                source_data_age_seconds=age_s,
                size_category=None,
                horizons=[],
            )
        )

    if not icebergs:
        detail = (
            f"skipped_coord={skipped_coord}, "
            f"skipped_date={skipped_date}, "
            f"skipped_domain={skipped_domain}"
        )
        raise UsnicParseError(
            f"USNIC CSV: no valid Antarctic iceberg rows found ({detail})"
        )

    # latest_timestamp is guaranteed non-None when icebergs is non-empty
    return icebergs, latest_timestamp  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_current_positions() -> tuple[list[IcebergForecast], datetime]:
    """
    Fetch current Antarctic iceberg positions from the USNIC weekly CSV.

    Returns
    -------
    (icebergs, as_of_timestamp)
      icebergs         — list of IcebergForecast with horizons=[] (positions
                         only; trajectory prediction is the ML model's job)
      as_of_timestamp  — the most recent 'Last Update' date in the returned
                         rows; all icebergs have lat < -40° (Southern Ocean)

    Raises
    ------
    UsnicNotFoundError  : HTTP 404 from USNIC
    UsnicNetworkError   : other network failure
    UsnicParseError     : CSV structure unexpected or no valid Antarctic rows
    """
    raw_bytes = _http_get(_USNIC_CSV_URL)
    return _parse_csv(raw_bytes)
