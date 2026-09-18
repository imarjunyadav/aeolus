"""
OSI SAF sea-ice connector (NRT and CDR).

Products and product naming (verified September 2026)
------------------------------------------------------
This connector targets the **SH multi-sensor blended NRT product**.

NRT multi-sensor (what this connector fetches):
  Filename: ice_conc_sh_polstere-100_multi_{YYYYMMDDHHMMSS}Z.nc
  Path    : /osisaf/met.no/ice/conc/{YYYY}/{MM}/
  Inputs  : sensor blend (historically SSMIS + AMSR; after September 2026
             primarily AMSR3/OSI-408-g).  The filename and THREDDS path
             are UNCHANGED regardless of which sensors feed the blend
             (confirmed from OSI SAF official announcement).

OSI-408-g single-sensor (AMSR3, NOT fetched by this connector):
  Filename: ice_conc_sh_polstere-100_amsr3_{YYYYMMDDHHMM}.nc
            Note: 12-digit timestamp (no seconds), no trailing "Z".
  FTP path: ftp://osisaf.met.no/prod/ice/conc_amsr
  THREDDS : UNCONFIRMED — likely /osisaf/met.no/ice/conc_amsr/{YYYY}/{MM}/
            but not verified from live catalogue (proxy blocks thredds.met.no).
  Status  : Operational from 1 September 2026.  A separate fetch function
            is needed once the THREDDS dodsC path is confirmed.

Note on OSI-401-d (SSMIS):
  Discontinued September 2026.  The multi-sensor product continues without
  SSMIS by using AMSR3 (OSI-408-g) as the primary input.

CDR (Climate Data Record):
  OSI-450-a  — 1978–2020 historical record.
  OSI-430-a  — ICDR (Interim CDR) 2021–present.

Source: MET Norway THREDDS — https://thredds.met.no/thredds/osisaf/
Auth  : Anonymous; no credentials required.

Variables retrieved
-------------------
ice_conc              [%]      Filtered sea ice concentration (0–100).
                               Stored as short int with scale_factor=0.01.
algorithm_uncertainty [%]      Uncertainty from the retrieval algorithm.
                               (Phase 2 doc incorrectly called this
                               "algorithm_standard_error" — the real variable
                               name is "algorithm_uncertainty".)
smearing_uncertainty  [%]      Uncertainty from temporal smearing.
total_uncertainty     [%]      Combined: sqrt(alg² + smear²).
                               Present as a pre-computed variable in current
                               NRT and CDR files.  If absent (e.g. older
                               CDR files), the connector derives it from
                               algorithm_uncertainty and smearing_uncertainty
                               using the same formula; the derived variable
                               carries long_name "total uncertainty (derived)".
status_flag           [bit]    Quality/screening flag (0–255 bit-coded).
                               Bit 0 = 0 → nominal; see product manual for
                               full bit definitions (Open water filter,
                               NWP skin temp, polarisation diff, ice
                               climatology, lake mask, land mask, coast,
                               missing value).

ice_conc, algorithm_uncertainty, smearing_uncertainty, and status_flag are
required in every returned dataset.  total_uncertainty is guaranteed present
either from the file or from the derivation step.

Grid (SH multi-sensor 10 km)
------------------------------
Projection : polar_stereographic, true scale at 70°S, origin at South Pole,
             central meridian 0°, WGS84 ellipsoid.
PROJ string: "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +datum=WGS84 +units=m"
Grid size  : 790 × 830 (xc × yc), 10 km × 10 km spacing.
Coord units: kilometres in the NetCDF file (multiply by 1000 for metres).

Timestamp semantics
-------------------
  source_data_timestamp — validity time decoded from ds["time"][0].
                          For the daily NRT product this is 12:00 UTC of
                          the analysis day.
  valid_at              — same as source_data_timestamp.
  fetch_time            — recorded internally (not exposed in the schema
                          return value; available in _fetch_meta()).

Failure taxonomy
----------------
  OsisafNetworkError(SeaIceConnectorError)
      Any network, proxy, HTTP, or connection error reaching THREDDS.

  OsisafNotFoundError(SeaIceConnectorError)
      File not in the THREDDS catalogue (404 or dataset does not exist).

  OsisafParseError(SeaIceConnectorError)
      Returned dataset missing expected variables or coordinates.

Network access note
-------------------
This connector makes outbound requests to thredds.met.no.  In environments
where that host is blocked (e.g. proxy 403), all public functions raise
OsisafNetworkError.  Integration tests skip gracefully when the host is
unreachable.

OPeNDAP engine
--------------
xarray opens the OPeNDAP URL with the pydap engine.  If pydap is not
installed all public functions raise OsisafNetworkError with an install
hint.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from cloud.data.exceptions import SeaIceConnectorError  # noqa: F401


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class OsisafNetworkError(SeaIceConnectorError):
    """Network, proxy, or HTTP failure reaching OSI SAF THREDDS."""


class OsisafNotFoundError(SeaIceConnectorError):
    """File or dataset not found in the THREDDS catalogue (HTTP 404)."""


class OsisafParseError(SeaIceConnectorError):
    """Retrieved dataset has unexpected structure or missing variables."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OPeNDAP base path on MET Norway THREDDS
_THREDDS_BASE = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/ice/conc"
_CDR_BASE     = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/ice/conc_cdr"
_ICDR_BASE    = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/ice/conc_cdrv3"

# NRT filename pattern — SH multi-sensor blended product.
# Datetime token: YYYYMMDDHHmmss (14 digits) + "Z"; reference time 12:00:00 UTC.
# This filename is UNCHANGED after the SSMIS→AMSR3 transition (confirmed).
_SH_NRT_PATTERN = "ice_conc_sh_polstere-100_multi_{dt}Z.nc"

# OSI-408-g single-sensor (AMSR3) filename — NOT fetched by this connector.
# Datetime token: YYYYMMDDHHmm (12 digits, no seconds, no trailing "Z").
# THREDDS dodsC path is UNCONFIRMED; implement once live catalogue is verified.
_SH_AMSR3_PATTERN = "ice_conc_sh_polstere-100_amsr3_{dt}.nc"  # {dt} = YYYYMMDDHHmm

_CDR_SH_PATTERN = "ice_conc_sh_polstere-100_reproc_{dt}Z.nc"

# OSI SAF SH polar-stereographic projection.
# Source: OSI SAF product user manual (OSI-401-d, 10 km SH grid).
# Coordinates in the NetCDF files are in km; multiply × 1000 for metres.
_SH_PROJ = "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +datum=WGS84 +units=m"

# Variable names (verified against current OSI SAF product files)
_VAR_ICE_CONC  = "ice_conc"
_VAR_ALG_UNC   = "algorithm_uncertainty"
_VAR_SMEAR_UNC = "smearing_uncertainty"
_VAR_TOTAL_UNC = "total_uncertainty"
_VAR_STATUS    = "status_flag"

_EXPECTED_VARS = {_VAR_ICE_CONC, _VAR_ALG_UNC, _VAR_SMEAR_UNC, _VAR_STATUS}

# Dimension names in OSI SAF SH files
_XC_DIM = "xc"
_YC_DIM = "yc"

# Tokens that indicate proxy / network blockage in error messages
_NETWORK_SKIP_TOKENS = (
    "403", "407", "proxy", "forbidden", "connection", "refused",
    "timeout", "resolve", "unreachable", "reset", "ssl", "certificate",
    "stdin", "credentials", "username", "password",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_nrt_url(target_date: date) -> str:
    """
    Construct the OPeNDAP URL for the SH NRT concentration file for
    a given analysis date.  The daily NRT product uses 12:00:00 UTC as
    its reference time.
    """
    dt_str = target_date.strftime("%Y%m%d") + "120000"
    filename = _SH_NRT_PATTERN.format(dt=dt_str)
    year  = target_date.year
    month = target_date.month
    return f"{_THREDDS_BASE}/{year}/{month:02d}/{filename}"


def _build_cdr_url(target_date: date) -> str:
    """Construct the OPeNDAP URL for the SH CDR/ICDR concentration file."""
    dt_str = target_date.strftime("%Y%m%d") + "120000"
    filename = _CDR_SH_PATTERN.format(dt=dt_str)
    year  = target_date.year
    month = target_date.month
    base = _ICDR_BASE if target_date.year >= 2021 else _CDR_BASE
    return f"{base}/{year}/{month:02d}/{filename}"


def _classify_error(exc: Exception, url: str = "") -> SeaIceConnectorError:
    """Map a low-level exception to our typed hierarchy."""
    msg = str(exc)
    lower = msg.lower()

    if "404" in msg or "not found" in lower or "no such file" in lower:
        return OsisafNotFoundError(
            f"OSI SAF file not found on THREDDS "
            f"(HTTP 404 or catalogue missing): {msg}"
        )

    if "401" in msg or "403" in msg or "407" in msg or "unauthorized" in lower:
        return OsisafNetworkError(
            f"OSI SAF THREDDS access denied (proxy or auth): {msg}"
        )

    if isinstance(exc, (OSError, ConnectionError)):
        return OsisafNetworkError(f"OSI SAF network/IO error: {msg}")

    return OsisafNetworkError(f"OSI SAF error ({type(exc).__name__}): {msg}")


def _open_opendap(url: str) -> Any:
    """
    Open an OPeNDAP URL as an xarray Dataset using the pydap engine.

    Raises OsisafNetworkError if pydap is not installed or any network
    error occurs; OsisafNotFoundError for HTTP 404.
    """
    try:
        import xarray as xr  # noqa: PLC0415
    except ImportError as exc:
        raise OsisafNetworkError(
            "xarray is not installed; run: pip install xarray"
        ) from exc

    try:
        import pydap  # noqa: F401, PLC0415 — existence check
    except ImportError as exc:
        raise OsisafNetworkError(
            "pydap is not installed; run: pip install pydap"
        ) from exc

    try:
        return xr.open_dataset(url, engine="pydap")
    except Exception as exc:
        raise _classify_error(exc, url) from exc


def _extract_validity_time(ds: Any) -> datetime | None:
    """
    Extract the model/analysis validity time from the dataset's ``time``
    coordinate (first element, xarray-decoded).

    Returns a UTC-aware datetime, or None if the coordinate is absent.
    """
    try:
        import numpy as np
        import pandas as pd
        t = pd.Timestamp(ds["time"].values[0]).to_pydatetime()
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t
    except (KeyError, IndexError, AttributeError, TypeError):
        return None


def _require_vars(ds: Any) -> None:
    """Raise OsisafParseError if any required variable is absent."""
    missing = _EXPECTED_VARS - set(ds.data_vars)
    if missing:
        raise OsisafParseError(
            f"OSI SAF dataset missing expected variable(s): "
            f"{sorted(missing)}.  Present: {sorted(ds.data_vars)}"
        )


def _derive_total_uncertainty(ds: Any) -> Any:
    """
    Ensure ``total_uncertainty`` is present in the dataset.

    Current OSI SAF NRT and CDR files include ``total_uncertainty`` as a
    pre-computed field: sqrt(algorithm_uncertainty² + smearing_uncertainty²).
    This function is a safety net for older or non-standard files where the
    pre-computed variable is absent.

    Behaviour
    ---------
    * If ``total_uncertainty`` is already in ``ds.data_vars``: returns ``ds``
      unchanged (the pre-computed value is used as-is).
    * If it is absent: computes sqrt(alg² + smear²) from the two component
      fields (both are guaranteed present by ``_require_vars``) and returns a
      new Dataset with ``total_uncertainty`` added.  The derived variable
      carries ``long_name = "total uncertainty (derived)"`` so callers can
      distinguish the source.

    Returns
    -------
    xarray.Dataset with ``total_uncertainty`` guaranteed present.
    """
    import numpy as np  # noqa: PLC0415

    if _VAR_TOTAL_UNC in ds.data_vars:
        return ds

    # Verify both components are present before attempting derivation.
    missing = [v for v in (_VAR_ALG_UNC, _VAR_SMEAR_UNC) if v not in ds.data_vars]
    if missing:
        raise OsisafParseError(
            f"Cannot derive '{_VAR_TOTAL_UNC}': required component(s) "
            f"{missing} are absent from the dataset and "
            f"'{_VAR_TOTAL_UNC}' is not pre-computed in the source file."
        )

    alg   = ds[_VAR_ALG_UNC].astype(float)
    smear = ds[_VAR_SMEAR_UNC].astype(float)
    total = (alg ** 2 + smear ** 2) ** 0.5
    total.attrs = {
        "units": "%",
        "long_name": "total uncertainty (derived)",
        "comment": (
            "Derived by this connector as sqrt(algorithm_uncertainty^2 + "
            "smearing_uncertainty^2); pre-computed total_uncertainty was "
            "absent in the source file."
        ),
    }
    return ds.assign({_VAR_TOTAL_UNC: total})


_COORD_UNITS_KM = {"km", "kilometers", "kilometres"}
_COORD_UNITS_M  = {"m", "meters", "metres"}
_COORD_UNITS_KNOWN = _COORD_UNITS_KM | _COORD_UNITS_M


def _normalise_coords(ds: Any) -> Any:
    """
    Ensure xc/yc coordinates are in metres.

    OSI SAF NRT files store xc/yc in kilometres (units='km').  This
    function converts them to metres in a copy of the Dataset so that
    regrid_polarstereo() — which expects metres — receives correct inputs.

    Accepted units (case-insensitive):
      'm', 'meters', 'metres'    → no change
      'km', 'kilometers', 'kilometres' → multiply × 1000

    Any other value (including an absent or empty units attribute) raises
    OsisafParseError so that silently wrong coordinates do not propagate
    into the regrid path.

    Parameters
    ----------
    ds : xarray.Dataset with xc/yc coordinates.

    Returns
    -------
    xarray.Dataset with xc/yc guaranteed in metres.

    Raises
    ------
    OsisafParseError : units attribute absent, empty, or unrecognised.
    """
    for dim in (_XC_DIM, _YC_DIM):
        if dim not in ds.coords:
            continue
        coord = ds[dim]
        units = coord.attrs.get("units", "").lower().strip()

        if not units:
            raise OsisafParseError(
                f"OSI SAF coordinate '{dim}' has no 'units' attribute; "
                f"cannot safely convert to metres for regrid.  "
                f"Expected one of: {sorted(_COORD_UNITS_KNOWN)}"
            )

        if units not in _COORD_UNITS_KNOWN:
            raise OsisafParseError(
                f"OSI SAF coordinate '{dim}' has unrecognised units "
                f"'{units}'; cannot safely convert to metres.  "
                f"Expected one of: {sorted(_COORD_UNITS_KNOWN)}"
            )

        if units in _COORD_UNITS_KM:
            ds = ds.assign_coords({dim: coord * 1000.0})
            ds[dim].attrs = {**coord.attrs, "units": "m"}

    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_nrt_concentration(target_date: date | None = None) -> object:
    """
    Fetch the SH NRT sea-ice concentration field from OSI SAF THREDDS.

    Uses OPeNDAP (pydap engine) to open the daily-mean file for
    ``target_date`` without downloading the full file.  Returns the lazy
    xarray Dataset on the native 10 km polar-stereographic grid so the
    caller can apply regrid_to_package_grid() independently.

    Parameters
    ----------
    target_date : date, optional
        Analysis date to fetch.  Defaults to yesterday UTC (most recent
        complete daily NRT field).

    Returns
    -------
    xarray.Dataset
        Variables: ``ice_conc`` (%, decoded floats after applying
        scale_factor), ``algorithm_uncertainty`` (%),
        ``smearing_uncertainty`` (%),
        ``total_uncertainty`` (%, from file or derived),
        ``status_flag`` (bit-coded short).
        Coordinates: ``xc`` and ``yc`` in metres (converted from km if
        needed), ``time`` (datetime64).
        ``total_uncertainty`` is always present: sourced directly from
        the file when available, otherwise derived as
        sqrt(algorithm_uncertainty² + smearing_uncertainty²) with
        long_name "total uncertainty (derived)".
        The dataset carries the native polar-stereographic grid;
        use regrid_to_package_grid(ds, target, "ice_conc",
        source_crs=SH_PROJ) to regrid.

    Raises
    ------
    OsisafNetworkError  : proxy, connection, or pydap missing.
    OsisafNotFoundError : file absent on THREDDS (wrong date or product
                          discontinued).
    OsisafParseError    : dataset missing required variables or coordinates.

    Note
    ----
    This function targets the multi-sensor blended NRT product.  The
    multi-sensor filename (``multi`` token) is unchanged after the SSMIS
    discontinuation; AMSR3 (OSI-408-g) feeds the blend from September 2026.
    For the OSI-408-g single-sensor product (amsr3 token, different THREDDS
    path, 12-digit timestamp) a separate fetch function is needed once the
    live THREDDS catalogue path is confirmed.
    """
    if target_date is None:
        target_date = (datetime.now(tz=timezone.utc) - timedelta(days=1)).date()

    url = _build_nrt_url(target_date)

    ds = _open_opendap(url)

    _require_vars(ds)
    ds = _normalise_coords(ds)
    ds = _derive_total_uncertainty(ds)

    validity_time = _extract_validity_time(ds)
    if validity_time is None:
        raise OsisafParseError(
            "OSI SAF dataset missing 'time' coordinate; "
            "cannot determine validity timestamp"
        )

    return ds


def fetch_cdr_concentration(target_date: date) -> object:
    """
    Fetch a historical CDR/ICDR sea-ice concentration field for a
    specific date.

    Routing:
      1978–2020 → OSI-450-a CDR  (``_CDR_BASE``)
      2021+     → OSI-430-a ICDR (``_ICDR_BASE``)

    Returns an xarray Dataset with the same variables and coordinate
    convention as :func:`fetch_nrt_concentration`.

    Raises
    ------
    OsisafNetworkError  : network or OPeNDAP error.
    OsisafNotFoundError : file absent on THREDDS.
    OsisafParseError    : dataset missing expected variables.
    """
    url = _build_cdr_url(target_date)

    ds = _open_opendap(url)

    _require_vars(ds)
    ds = _normalise_coords(ds)
    ds = _derive_total_uncertainty(ds)

    validity_time = _extract_validity_time(ds)
    if validity_time is None:
        raise OsisafParseError(
            "OSI SAF CDR dataset missing 'time' coordinate"
        )

    return ds


# ---------------------------------------------------------------------------
# Convenience: projection string exposed for callers using regrid
# ---------------------------------------------------------------------------

#: PROJ string for the OSI SAF SH 10 km polar-stereographic grid.
#: Pass as ``source_crs`` to cloud.data.regrid.regrid_to_package_grid().
SH_PROJ = _SH_PROJ
