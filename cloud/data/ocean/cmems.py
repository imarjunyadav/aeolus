"""
CMEMS ocean connector — Global Ocean Physics Analysis and Forecast (PHY 001_024).

Product : GLOBAL_ANALYSISFORECAST_PHY_001_024
Dataset : cmems_mod_glo_phy_anfc_0.083deg_P1D-m
Source  : https://marine.copernicus.eu
Auth    : Free CMEMS account.  copernicusmarine 2.x checks, in order:
            1. Explicit username/password kwargs
            2. Env vars COPERNICUSMARINE_SERVICE_USERNAME / …_PASSWORD
            3. Credentials file ~/.copernicusmarine/.copernicusmarine-credentials
               (created by ``copernicusmarine login``)
            4. Legacy ~/.netrc  or ~/motuclient/motuclient-python.ini

Spatial resolution  : 1/12° ≈ 8 km
Temporal resolution : daily mean (P1D)
Coverage            : 80°S – 90°N global, 10-day forecast updated daily

Variables retrieved
-------------------
uo      [m/s]   eastward sea water velocity (surface, depth ≈ 0.49 m)
vo      [m/s]   northward sea water velocity
thetao  [°C]    sea water potential temperature (proxy for SST at surface)
siconc  [1]     sea ice area fraction (0–1)
sithick [m]     sea ice thickness
mlotst  [m]     ocean mixed layer thickness (sigma_theta criterion)

All fields are returned at a single surface level (depth index 0).
Requesting only the subset needed keeps download volumes small (<1 MB for a
20°×20° Antarctic region, one day).

Phase 3 ingestion note
----------------------
OceanSnapshot is a Phase 1 schema designed to be area-representative: one
spatial-mean value per variable for the whole requested bounding box.  This is
sufficient for Phase 3 iceberg drift physics features (background current speed
and SST estimates for melt rate).  It is NOT the final spatial ocean
representation — Phase 4+ will supply full gridded current and temperature
fields required by the iceberg ML model and navigation risk engine.

Timestamp semantics
-------------------
  source_data_timestamp — the CMEMS model validity time extracted from the
      xarray dataset's ``time`` coordinate (first time step of the returned
      slice).  This is when the data is valid, not when it was downloaded.
  valid_at              — same as source_data_timestamp for this product
      (daily mean: the data is valid at the centre of the validity day).
  source_data_age_seconds — seconds elapsed between source_data_timestamp and
      the moment the fetch completed (fetch_time − source_data_timestamp).
  fetch_time            — internal only; stored in metadata["fetch_time_utc"]
      for diagnostics but not surfaced in the main schema fields.

Failure taxonomy
----------------
All public functions raise a subclass of OceanConnectorError:

  CmemsAuthError(OceanConnectorError)
      Credentials absent, invalid, or the authentication service is
      unreachable.  Covers: CredentialsCannotBeNone, InvalidUsernameOrPassword,
      CouldNotConnectToAuthenticationSystem.

  CmemsNotFoundError(OceanConnectorError)
      Product or dataset not found.  Covers: ProductNotFound, DatasetNotFound,
      DatasetVersionNotFound.

  CmemsNetworkError(OceanConnectorError)
      Any other network or HTTP failure (proxy, connection error, timeout).

  CmemsParseError(OceanConnectorError)
      The returned xarray Dataset is missing expected variables or coordinates,
      or cannot be coerced into OceanSnapshot values.

Network access note
-------------------
This connector makes real outbound requests to stac.marine.copernicus.eu and
s3.waw3-1.cloudferro.com.  In environments where those hosts are blocked (e.g.
proxy policy 403), all public functions raise CmemsNetworkError.  Real-data
verification was not possible in the CCR build environment; integration tests
skip with the proxy error message when the host is unreachable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from cloud.data.exceptions import OceanConnectorError
from shared.schemas.forecast import OceanSnapshot


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class CmemsNetworkError(OceanConnectorError):
    """Network or proxy failure reaching CMEMS (non-auth, non-404)."""


class CmemsAuthError(OceanConnectorError):
    """Invalid, missing, or unreachable CMEMS credentials."""


class CmemsNotFoundError(OceanConnectorError):
    """Product or dataset not found in the CMEMS catalogue."""


class CmemsParseError(OceanConnectorError):
    """Retrieved dataset has unexpected structure or missing variables."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRODUCT_ID  = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
_DATASET_ID  = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
_SURFACE_DEPTH_M = 0.49   # nominal depth of the first model level

# Variables we request; all live in the same dataset.
_VARIABLES = ["uo", "vo", "thetao", "siconc", "sithick", "mlotst"]

# Mapping of copernicusmarine exception names to our hierarchy (by class name).
# Checked at runtime to avoid hard import coupling to internal exception classes.
_AUTH_EXCEPTIONS = {
    "CredentialsCannotBeNone",
    "InvalidUsernameOrPassword",
    "CouldNotConnectToAuthenticationSystem",
}
_NOTFOUND_EXCEPTIONS = {
    "ProductNotFound",
    "DatasetNotFound",
    "DatasetVersionNotFound",
    "DatasetVersionPartNotFound",
    "VariableDoesNotExistInTheDataset",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_cmems_error(exc: Exception) -> OceanConnectorError:
    """
    Map a copernicusmarine exception to our typed hierarchy.

    We match by class name rather than by direct isinstance() to avoid
    tightly coupling this module to copernicusmarine internals that may change
    across library versions.
    """
    cls_name = type(exc).__name__
    msg = str(exc)

    if cls_name in _AUTH_EXCEPTIONS:
        return CmemsAuthError(f"CMEMS authentication failed ({cls_name}): {msg}")

    if cls_name in _NOTFOUND_EXCEPTIONS:
        return CmemsNotFoundError(f"CMEMS product/dataset not found ({cls_name}): {msg}")

    # Connection / proxy / SSL errors are typically OSError or subclasses.
    if isinstance(exc, (OSError, ConnectionError)):
        return CmemsNetworkError(f"CMEMS network error: {msg}")

    # Anything else from the library treated as network (catch-all).
    return CmemsNetworkError(f"CMEMS error ({cls_name}): {msg}")


def _scalar(ds: Any, var: str) -> float | None:
    """
    Extract a scalar float from an xarray Dataset variable.

    Averages over all spatial dimensions at the first time step.
    Returns None if the variable is absent or contains only NaN.
    """
    try:
        import numpy as np
        da = ds[var]
        # Drop depth if present; take surface (index 0).
        if "depth" in da.dims:
            da = da.isel(depth=0)
        # Take first time step.
        if "time" in da.dims:
            da = da.isel(time=0)
        # Spatial mean ignoring NaN.
        vals = da.values.astype(float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return None
        return float(finite.mean())
    except (KeyError, IndexError, AttributeError):
        return None


def _extract_model_time(ds: Any) -> datetime | None:
    """
    Extract the model validity time from the xarray Dataset's ``time``
    coordinate (first time step of the returned slice).

    Returns a UTC-aware datetime, or None if the coordinate is absent or
    cannot be parsed.
    """
    try:
        import numpy as np
        import pandas as pd
        time_vals = ds["time"].values
        # numpy datetime64 → pandas Timestamp → python datetime
        t = pd.Timestamp(time_vals[0]).to_pydatetime()
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t
    except (KeyError, IndexError, AttributeError, TypeError):
        return None


def _ds_to_snapshot(ds: Any, fetch_time: datetime) -> OceanSnapshot:
    """
    Convert an xarray Dataset (from copernicusmarine.open_dataset) into an
    OceanSnapshot.  Only the surface-level spatial mean is retained.

    Timestamp semantics (see module docstring for full explanation):
      source_data_timestamp  — CMEMS model validity time from ds["time"][0]
      valid_at               — same as source_data_timestamp
      source_data_age_seconds — fetch_time − source_data_timestamp (seconds)
      fetch_time             — stored in metadata["fetch_time_utc"] only

    Raises CmemsParseError if neither current component (uo/vo) can be read,
    or if the model validity time cannot be extracted from the dataset.
    """
    model_time = _extract_model_time(ds)
    if model_time is None:
        raise CmemsParseError(
            "CMEMS dataset missing 'time' coordinate; "
            "cannot determine model validity timestamp"
        )

    uo     = _scalar(ds, "uo")
    vo     = _scalar(ds, "vo")
    thetao = _scalar(ds, "thetao")
    siconc = _scalar(ds, "siconc")
    # sithick and mlotst may be absent in some dataset parts.
    sithick = _scalar(ds, "sithick")
    mlotst  = _scalar(ds, "mlotst")

    if uo is None and vo is None:
        raise CmemsParseError(
            "CMEMS dataset contains no valid surface current values "
            "(both uo and vo are absent or all-NaN)"
        )

    available: list[str] = []
    for name, val in [("uo", uo), ("vo", vo), ("thetao", thetao),
                      ("siconc", siconc), ("sithick", sithick), ("mlotst", mlotst)]:
        if val is not None:
            available.append(name)

    age_s = max(0, int((fetch_time - model_time).total_seconds()))

    return OceanSnapshot(
        source_data_timestamp=model_time,
        source_data_age_seconds=age_s,
        valid_at=model_time,
        current_u_ms=uo,
        current_v_ms=vo,
        sst_c=thetao,
        available_variables=available,
        metadata={
            "product_id": _PRODUCT_ID,
            "dataset_id": _DATASET_ID,
            "fetch_time_utc": fetch_time.isoformat(),
            "siconc": siconc,
            "sithick": sithick,
            "mlotst": mlotst,
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ocean_snapshot(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    target_date: date | None = None,
    username: str | None = None,
    password: str | None = None,
) -> OceanSnapshot:
    """
    Fetch a spatial-mean OceanSnapshot for the given bounding box from CMEMS.

    Retrieves a single daily-mean layer (the most recent analysis or the
    analysis for ``target_date`` if specified) for the CMEMS PHY 001_024
    product.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds in signed decimal degrees (negative = south).
    lon_min, lon_max : float
        Longitude bounds in signed decimal degrees (negative = west).
    target_date : date, optional
        The calendar day to fetch.  Defaults to yesterday UTC (last completed
        daily mean).
    username, password : str, optional
        Explicit CMEMS credentials.  If omitted the library reads
        ``~/.copernicusmarine/.copernicusmarine-credentials`` or the
        environment variables ``COPERNICUSMARINE_SERVICE_USERNAME`` /
        ``COPERNICUSMARINE_SERVICE_PASSWORD``.

    Returns
    -------
    OceanSnapshot
        Spatial-mean values over the requested bounding box.

    Raises
    ------
    CmemsAuthError     : credentials absent or rejected
    CmemsNotFoundError : product or dataset not in CMEMS catalogue
    CmemsNetworkError  : proxy, connection, or other network failure
    CmemsParseError    : returned dataset missing expected variables
    """
    try:
        import copernicusmarine  # deferred: not available in all envs
    except ImportError as exc:
        raise CmemsNetworkError(
            "copernicusmarine package is not installed; "
            "run: pip install copernicusmarine>=2.0"
        ) from exc

    if target_date is None:
        target_date = (datetime.now(tz=timezone.utc) - timedelta(days=1)).date()

    start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
    end   = start + timedelta(hours=23, minutes=59, seconds=59)

    fetch_time = datetime.now(tz=timezone.utc)

    open_kwargs: dict[str, Any] = dict(
        dataset_id=_DATASET_ID,
        variables=_VARIABLES,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        minimum_depth=0.0,
        maximum_depth=_SURFACE_DEPTH_M,
        start_datetime=start,
        end_datetime=end,
    )
    if username is not None:
        open_kwargs["username"] = username
    if password is not None:
        open_kwargs["password"] = password

    try:
        ds = copernicusmarine.open_dataset(**open_kwargs)
    except Exception as exc:
        classified = _classify_cmems_error(exc)
        raise classified from exc

    try:
        return _ds_to_snapshot(ds, fetch_time)
    except CmemsParseError:
        raise
    except Exception as exc:
        raise CmemsParseError(
            f"CMEMS dataset could not be converted to OceanSnapshot: {exc}"
        ) from exc
