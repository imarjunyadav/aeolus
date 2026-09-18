"""
NSIDC G02135 sea ice connector — Southern Hemisphere daily extent + concentration.

Product : Sea Ice Index v4 (G02135)
Source  : https://noaadata.apps.nsidc.org/NOAA/G02135/
Auth    : None required.
Use     : Fallback when OSI SAF is unavailable; extent/anomaly baseline in all modes.

Failure taxonomy
----------------
All public functions raise a subclass of SeaIceConnectorError:

  NsidcNotFoundError(NsidcNetworkError)
      HTTP 404 — the requested date is not yet available (data lags ~1-3 days)
      or the URL structure has changed.

  NsidcNetworkError(SeaIceConnectorError)
      Any other network or HTTP failure: connection refused, proxy denial,
      timeout, non-200 status code.

  NsidcParseError(SeaIceConnectorError)
      The file was fetched successfully but could not be parsed: unexpected
      CSV structure, missing columns, unreadable GeoTIFF, scaling outside
      valid range, or unexpected raw pixel values outside the documented
      valid/flag ranges.

These are deliberately separate so callers can distinguish:
  - "date not ready yet, retry tomorrow"  → NsidcNotFoundError
  - "NSIDC unreachable, fall back to OSI SAF or cached data"  → NsidcNetworkError
  - "data file is malformed, alert operator"  → NsidcParseError

Products fetched
----------------
  Extent CSV  (cumulative, 1978-present):
    .../south/daily/csv/S_seaice_extent_daily_v3.0.csv
    Returns: {'date': date, 'extent_mkm2': float, 'area_mkm2': float,
              'source_data': str}  — extent/area in millions of km².

  Concentration GeoTIFF (daily, ~25 km):
    .../south/daily/geotiff/{YYYY}/{MM}_{Mon}/S_{YYYYMMDD}_concentration_v3.0.tif
    Returns: xarray.DataArray, dims ['y', 'x'], coordinates in EPSG:3031 metres.
    Values 0.0–1.0 (ice fraction). Special flags (land/coast/missing) → NaN.
    CRS stored in attrs['crs'] as PROJ string.

GeoTIFF pixel encoding (G02135 v3 User Guide, Table 5)
-------------------------------------------------------
  Valid concentration : 0–1000   → divide by 1000.0 to get fraction 0.0–1.0
  Known special flags (masked to NaN):
    2510  coast_line   (dark gray)
    2530  land         (black)
    2540  missing      (gray)
    2550  no_data      (yellow)
  Any other value (1001–2509, 2511–2529, 2531–2539, 2541–2549, > 2550,
  or any negative) is unexpected and raises NsidcParseError.

Network access note
-------------------
  This connector makes real outbound requests to noaadata.apps.nsidc.org.
  In environments where that host is blocked (e.g. proxy policy 403), all
  public functions raise NsidcNetworkError.  Real-data verification was not
  possible in the CCR build environment; integration tests skip with the
  proxy error message when the host is unreachable.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from calendar import month_abbr
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

from cloud.data.exceptions import SeaIceConnectorError


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class NsidcNetworkError(SeaIceConnectorError):
    """Network or HTTP failure reaching NSIDC (non-404)."""


class NsidcNotFoundError(NsidcNetworkError):
    """HTTP 404 — date not yet available or URL structure changed."""


class NsidcParseError(SeaIceConnectorError):
    """Downloaded content could not be parsed."""


# ---------------------------------------------------------------------------
# URL constants
# ---------------------------------------------------------------------------

_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily"
_EXTENT_CSV_URL = f"{_BASE}/csv/S_seaice_extent_daily_v3.0.csv"
_TIFF_URL_TMPL = "{base}/geotiff/{yyyy}/{mm:02d}_{mon}/S_{yyyymmdd}_concentration_v3.0.tif"

# Valid concentration range: raw 0–1000 → fraction 0.0–1.0 (divide by _SCALE_FACTOR).
_VALID_CONC_MAX = 1000
_SCALE_FACTOR = 1000.0

# Known special-flag pixel values from G02135 v3 User Guide Table 5.
# Any raw integer value not in 0–_VALID_CONC_MAX and not in this set is
# unexpected/corrupt and will raise NsidcParseError.
_KNOWN_FLAGS: dict[int, str] = {
    2510: "coast_line",
    2530: "land",
    2540: "missing",
    2550: "no_data",
}

_HTTP_TIMEOUT_S = 30
_USER_AGENT = "aeolus-nsidc-connector/1.0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _concentration_url(d: date) -> str:
    """Build the GeoTIFF URL for a given date."""
    mon = month_abbr[d.month]  # 'Jan' … 'Dec'
    return _TIFF_URL_TMPL.format(
        base=_BASE,
        yyyy=d.year,
        mm=d.month,
        mon=mon,
        yyyymmdd=d.strftime("%Y%m%d"),
    )


def _http_get(url: str) -> bytes:
    """
    Fetch URL bytes.

    Raises
    ------
    NsidcNotFoundError  : HTTP 404
    NsidcNetworkError   : any other HTTP error or connection failure
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise NsidcNotFoundError(
                f"NSIDC 404 — {url!r}: date not yet available or URL changed"
            ) from exc
        raise NsidcNetworkError(
            f"NSIDC HTTP {exc.code} fetching {url!r}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise NsidcNetworkError(
            f"NSIDC network error fetching {url!r}: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise NsidcNetworkError(
            f"NSIDC I/O error fetching {url!r}: {exc}"
        ) from exc


def _parse_extent_csv(raw_text: str, target: date) -> dict[str, Any]:
    """
    Parse the cumulative daily-extent CSV and return the row for `target`.

    Expected format (skipping comment/header lines):
        Year,  Mo,  Day,       Extent,     Area,  Missing,Source Data, hemisphere
        1978,  10,  26,      17.006,    14.569,    0,NSIDC-0051, S

    Raises
    ------
    NsidcParseError : CSV has unexpected structure or target date not found.
    """
    import csv

    lines = raw_text.splitlines()
    # Skip leading non-data lines (comments like "Updating …" and blank lines)
    data_lines: list[str] = []
    header_line: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # The column header row contains "year" (case-insensitive).
        # Skip other non-digit lines (e.g. "Updating …" comment).
        if header_line is None and not stripped[0].isdigit() and "year" in stripped.lower():
            header_line = stripped
            continue
        if header_line is not None and stripped[0].isdigit():
            data_lines.append(stripped)

    if header_line is None:
        raise NsidcParseError("NSIDC extent CSV: no header row found")

    # Normalise header field names (strip whitespace)
    raw_headers = [h.strip().lower() for h in header_line.split(",")]
    # Expected: year, mo, day, extent, area, missing, source data, hemisphere
    required = {"year", "mo", "day", "extent", "area"}
    missing_cols = required - set(raw_headers)
    if missing_cols:
        raise NsidcParseError(
            f"NSIDC extent CSV: missing columns {missing_cols}; got {raw_headers}"
        )

    idx_year = raw_headers.index("year")
    idx_mo   = raw_headers.index("mo")
    idx_day  = raw_headers.index("day")
    idx_ext  = raw_headers.index("extent")
    idx_area = raw_headers.index("area")
    idx_src  = raw_headers.index("source data") if "source data" in raw_headers else None

    for row_str in data_lines:
        cols = [c.strip() for c in row_str.split(",")]
        if len(cols) <= max(idx_year, idx_mo, idx_day, idx_ext, idx_area):
            continue
        try:
            yr  = int(cols[idx_year])
            mo  = int(cols[idx_mo])
            dy  = int(cols[idx_day])
        except ValueError:
            continue
        if date(yr, mo, dy) != target:
            continue

        # Found the row — parse extent and area
        try:
            extent = float(cols[idx_ext])
            area   = float(cols[idx_area])
        except ValueError as exc:
            raise NsidcParseError(
                f"NSIDC extent CSV: could not parse extent/area for {target}: "
                f"row={row_str!r}"
            ) from exc

        if extent == -9999:
            raise NsidcParseError(
                f"NSIDC extent CSV: missing-data sentinel (-9999) for {target}"
            )

        src = cols[idx_src].strip() if idx_src is not None and idx_src < len(cols) else "unknown"
        return {
            "date": target,
            "extent_mkm2": extent,
            "area_mkm2":   area,
            "source_data": src,
        }

    raise NsidcParseError(
        f"NSIDC extent CSV: no row found for {target} "
        f"(file covers {len(data_lines)} days)"
    )


def _parse_concentration_tiff(raw_bytes: bytes) -> object:
    """
    Decode a GeoTIFF into an xarray.DataArray on the native EPSG:3031 grid.

    Values 0–1000 raw → 0.0–1.0 fraction.
    Land / coast / lake / missing flags (> 1000) → NaN.

    Returns xarray.DataArray with dims ['y', 'x'] and coords in metres.

    Raises
    ------
    NsidcParseError  : rasterio cannot open the bytes, or the data has
                       unexpected shape / dtype.
    ImportError      : rasterio is not installed.
    """
    import xarray as xr

    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for NSIDC GeoTIFF reading: "
            "pip install rasterio"
        )

    try:
        with rasterio.open(io.BytesIO(raw_bytes)) as ds:
            raw_data = ds.read(1).astype(float)    # shape (height, width)
            transform = ds.transform
            crs_str   = ds.crs.to_string() if ds.crs else "EPSG:3031"
            height, width = ds.height, ds.width
    except Exception as exc:
        raise NsidcParseError(
            f"NSIDC concentration GeoTIFF: rasterio could not open file: {exc}"
        ) from exc

    # Build x/y coordinate arrays from the affine transform.
    # transform maps (col, row) → (x, y); pixel centres at +0.5 offset.
    xs = np.array(
        [transform.c + (j + 0.5) * transform.a for j in range(width)],
        dtype=float,
    )
    ys = np.array(
        [transform.f + (i + 0.5) * transform.e for i in range(height)],
        dtype=float,
    )

    # Three-way pixel classification per G02135 v3 User Guide Table 5:
    #   valid        : integer value in [0, _VALID_CONC_MAX]
    #   known flag   : integer value in _KNOWN_FLAGS → mask to NaN
    #   unexpected   : anything else → corrupt data, raise NsidcParseError
    raw_int = raw_data.astype(np.int32)
    valid_mask    = (raw_int >= 0) & (raw_int <= _VALID_CONC_MAX)
    known_flag_vals = np.array(list(_KNOWN_FLAGS.keys()), dtype=np.int32)
    flag_mask     = np.isin(raw_int, known_flag_vals)
    unexpected_mask = ~valid_mask & ~flag_mask

    if unexpected_mask.any():
        unexpected_vals = np.unique(raw_int[unexpected_mask]).tolist()
        raise NsidcParseError(
            f"NSIDC concentration GeoTIFF: unexpected raw pixel values "
            f"{unexpected_vals}; expected 0–{_VALID_CONC_MAX} (concentration) "
            f"or known flags {list(_KNOWN_FLAGS.keys())}"
        )

    # Mask known flags to NaN, scale valid values to 0.0–1.0
    raw_data[flag_mask] = np.nan
    concentration = raw_data / _SCALE_FACTOR

    return xr.DataArray(
        concentration,
        dims=["y", "x"],
        coords={"y": ys, "x": xs},
        attrs={
            "crs":          crs_str,
            "source":       "NSIDC G02135",
            "units":        "fraction (0.0–1.0)",
            "raw_scale":    f"raw / {int(_SCALE_FACTOR)}",
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_extent_csv(target_date: date | None = None) -> dict[str, Any]:
    """
    Fetch the Southern Hemisphere sea ice extent for a given date.

    Parameters
    ----------
    target_date : date to query; defaults to yesterday UTC if None.

    Returns
    -------
    dict with keys: date, extent_mkm2, area_mkm2, source_data.

    Raises
    ------
    NsidcNotFoundError  : HTTP 404 from NSIDC
    NsidcNetworkError   : other network failure
    NsidcParseError     : CSV structure unexpected or date absent
    """
    if target_date is None:
        target_date = datetime.now(tz=timezone.utc).date()

    raw_bytes = _http_get(_EXTENT_CSV_URL)
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NsidcParseError(f"NSIDC extent CSV is not UTF-8: {exc}") from exc

    return _parse_extent_csv(raw_text, target_date)


def fetch_concentration_geotiff(target_date: date | None = None) -> object:
    """
    Fetch the daily Southern Hemisphere concentration GeoTIFF.

    Parameters
    ----------
    target_date : date of the product; defaults to yesterday UTC if None.

    Returns
    -------
    xarray.DataArray, dims ['y', 'x'], coords in EPSG:3031 metres.
    Values 0.0–1.0 (ice fraction). Special flags → NaN.
    CRS is stored in attrs['crs'].

    Raises
    ------
    NsidcNotFoundError  : HTTP 404 (date not yet available)
    NsidcNetworkError   : other network failure
    NsidcParseError     : GeoTIFF could not be parsed
    """
    if target_date is None:
        target_date = datetime.now(tz=timezone.utc).date()

    url = _concentration_url(target_date)
    raw_bytes = _http_get(url)
    return _parse_concentration_tiff(raw_bytes)
