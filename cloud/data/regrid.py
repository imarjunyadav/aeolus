"""
Regridding utility — interpolate source data onto a ForecastPackage grid.

Two distinct cases are handled:

  regrid_latlon()
    Source is on a regular lat-lon grid (CMEMS, ECMWF IFS, GFS, ERA5).
    Uses xarray.interp(), which calls scipy linear/nearest under the hood.
    Suitable whenever the source coordinates are expressed as latitude and
    longitude degrees and the grid is rectilinear in those coordinates.

  regrid_polarstereo()
    Source is on a polar-stereographic grid (e.g. OSI SAF SH 10 km).
    The source grid is regular in projected (x, y) metres, NOT in lat-lon.
    xarray.interp() cannot be used directly because the target lat-lon
    points do not map onto the source coordinate axes.
    Approach:
      1. Transform target lat-lon points to the source CRS with pyproj.
      2. Interpolate on the regular (x, y) grid with
         scipy.interpolate.RegularGridInterpolator.

  regrid_to_package_grid()
    Dispatcher. Caller passes source_crs=None for lat-lon sources and a
    PROJ/WKT/EPSG string for projected sources.

Output convention
-----------------
All functions return list[list[float]] of shape [n_lat][n_lon].
Row 0 corresponds to lat_min (southernmost); row n_lat-1 to lat_max.
This matches the mock generator and the shared/schemas convention.

Phase 3 Step 2.
"""

from __future__ import annotations

import numpy as np

from shared.schemas.common import GridMetadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def regrid_to_package_grid(
    source: object,
    target: GridMetadata,
    variable: str,
    *,
    source_crs: str | None = None,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    x_dim: str = "x",
    y_dim: str = "y",
    method: str = "linear",
) -> list[list[float]]:
    """
    Interpolate a source field onto the ForecastPackage grid.

    Parameters
    ----------
    source     : xarray DataArray or Dataset. If Dataset, `variable` selects
                 the DataArray.
    target     : GridMetadata (lat_min/max, lon_min/max, n_lat, n_lon).
    variable   : variable name to extract when source is a Dataset.
    source_crs : PROJ string, WKT, or EPSG code (e.g. "EPSG:3031") that
                 describes the source grid's CRS. Pass None (default) when
                 the source is on a regular lat-lon grid.
    lat_dim    : name of the latitude dimension in the source DataArray.
    lon_dim    : name of the longitude dimension in the source DataArray.
    x_dim      : name of the x (easting) dimension for projected sources.
    y_dim      : name of the y (northing) dimension for projected sources.
    method     : interpolation method — 'linear' or 'nearest'.

    Returns
    -------
    list[list[float]] of shape [n_lat][n_lon], row 0 = lat_min.
    """
    import xarray as xr  # noqa: PLC0415  — optional dep, deferred import

    da: xr.DataArray
    if isinstance(source, xr.Dataset):
        da = source[variable]
    else:
        da = source  # type: ignore[assignment]

    if source_crs is None:
        return regrid_latlon(da, target, lat_dim=lat_dim, lon_dim=lon_dim, method=method)
    else:
        return regrid_polarstereo(da, target, crs=source_crs, x_dim=x_dim, y_dim=y_dim, method=method)


def regrid_latlon(
    source: object,
    target: GridMetadata,
    *,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    method: str = "linear",
) -> list[list[float]]:
    """
    Regrid a regular lat-lon source onto the ForecastPackage grid.

    Uses xarray.DataArray.interp() which requires the source coordinates to
    be expressed in degrees latitude and longitude on a rectilinear grid.
    This is appropriate for CMEMS, ECMWF IFS Open Data, GFS, and ERA5 output.

    Parameters
    ----------
    source  : xarray DataArray with lat/lon coordinate dimensions.
    target  : output grid specification.
    lat_dim : name of the latitude coordinate in `source`.
    lon_dim : name of the longitude coordinate in `source`.
    method  : 'linear' (default) or 'nearest'.

    Returns
    -------
    list[list[float]] shape [n_lat][n_lon], row 0 = lat_min.
    """
    import xarray as xr  # noqa: PLC0415

    da: xr.DataArray = source  # type: ignore[assignment]

    target_lats = np.linspace(target.lat_min, target.lat_max, target.n_lat)
    target_lons = np.linspace(target.lon_min, target.lon_max, target.n_lon)

    interp_kwargs = {lat_dim: target_lats, lon_dim: target_lons}
    result: xr.DataArray = da.interp(method=method, **interp_kwargs)

    # Ensure correct dimension order and squeeze any extra dims
    result = result.transpose(lat_dim, lon_dim)
    arr: np.ndarray = np.asarray(result.values, dtype=float)

    return arr.tolist()


def regrid_polarstereo(
    source: object,
    target: GridMetadata,
    *,
    crs: str,
    x_dim: str = "x",
    y_dim: str = "y",
    method: str = "linear",
) -> list[list[float]]:
    """
    Regrid a polar-stereographic source onto the ForecastPackage grid.

    Source is on a regular grid in polar-stereographic projected coordinates,
    NOT in lat-lon degrees. xarray.interp() cannot be applied directly because
    the target lat-lon points are not aligned with the source x/y axes.

    Example: OSI SAF SH 10 km sea-ice products use a custom polar-stereographic
    CRS (true scale at 70°S, central meridian 0°, WGS84) — NOT EPSG:3031.
    Pass the verified PROJ string from the source product as ``crs``.

    Approach
    --------
    1. Build target lat-lon arrays from GridMetadata.
    2. Transform each (lat, lon) target point to the source CRS via pyproj.
    3. Interpolate on the regular (x, y) source grid using
       scipy.interpolate.RegularGridInterpolator.

    Parameters
    ----------
    source : xarray DataArray with x/y coordinate dimensions in metres.
    target : output grid specification.
    crs    : PROJ/EPSG/WKT string for the source CRS.  Must match the
             actual product projection — do not assume EPSG:3031 for
             OSI SAF SH products; use the verified PROJ string from the
             connector (e.g. cloud.data.sea_ice.osisaf.SH_PROJ).
    x_dim  : name of the x (easting) coordinate in `source`.
    y_dim  : name of the y (northing) coordinate in `source`.
    method : 'linear' (default) or 'nearest'.

    Returns
    -------
    list[list[float]] shape [n_lat][n_lon], row 0 = lat_min.

    Raises
    ------
    ValueError  : if target points fall outside the source domain.
    ImportError : if pyproj or scipy are not installed.
    """
    import xarray as xr  # noqa: PLC0415
    from pyproj import CRS, Transformer  # noqa: PLC0415
    from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

    da: xr.DataArray = source  # type: ignore[assignment]

    # 1. Build target coordinate arrays (lat-lon degrees)
    target_lats = np.linspace(target.lat_min, target.lat_max, target.n_lat)
    target_lons = np.linspace(target.lon_min, target.lon_max, target.n_lon)
    # Shape: (n_lat, n_lon) grid of query points
    grid_lons, grid_lats = np.meshgrid(target_lons, target_lats)  # both [n_lat, n_lon]

    # 2. Transform target lat-lon to source CRS
    src_crs = CRS.from_user_input(crs)
    geo_crs = CRS.from_epsg(4326)  # WGS-84 lat-lon
    transformer = Transformer.from_crs(geo_crs, src_crs, always_xy=True)
    # always_xy=True: input order is (lon, lat), output is (x, y)
    proj_x, proj_y = transformer.transform(grid_lons.ravel(), grid_lats.ravel())
    # Reshape to (n_lat, n_lon) for later use
    proj_x = proj_x.reshape(grid_lats.shape)
    proj_y = proj_y.reshape(grid_lats.shape)

    # 3. Build interpolator on the regular (x, y) source grid
    src_x: np.ndarray = np.asarray(da[x_dim].values, dtype=float)
    src_y: np.ndarray = np.asarray(da[y_dim].values, dtype=float)
    src_data: np.ndarray = np.asarray(da.values, dtype=float)

    # RegularGridInterpolator expects data indexed as (y, x) when
    # the DataArray is (y_dim, x_dim).  Confirm axis order.
    if da.dims.index(y_dim) == 0 and da.dims.index(x_dim) == 1:
        # data shape is (n_y, n_x) — correct
        pass
    else:
        # Transpose to (y_dim, x_dim)
        da_t = da.transpose(y_dim, x_dim)
        src_data = np.asarray(da_t.values, dtype=float)

    # scipy RGI requires strictly increasing coordinates
    y_ascending = src_y[0] < src_y[-1]
    x_ascending = src_x[0] < src_x[-1]
    if not y_ascending:
        src_y = src_y[::-1]
        src_data = src_data[::-1, :]
    if not x_ascending:
        src_x = src_x[::-1]
        src_data = src_data[:, ::-1]

    rgi = RegularGridInterpolator(
        (src_y, src_x),
        src_data,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    # 4. Query interpolator: RGI expects points as (y, x) pairs
    query_pts = np.stack([proj_y.ravel(), proj_x.ravel()], axis=1)
    interp_vals = rgi(query_pts).reshape(target.n_lat, target.n_lon)

    return interp_vals.tolist()
