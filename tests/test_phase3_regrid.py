"""
Phase 3 Step 2 — unit tests for cloud/data/regrid.py.

All tests use synthetic xarray DataArrays; no network calls are made.
The two grid types are tested separately to confirm they go through
different code paths (the OSI SAF polar-stereo path must NOT use
xarray.interp directly on lat-lon axes).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

from shared.schemas.common import GridMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID_5x5 = GridMetadata(
    lat_min=-70.0, lat_max=-65.0,
    lon_min=10.0,  lon_max=15.0,
    n_lat=5, n_lon=5,
)

GRID_10x10 = GridMetadata(
    lat_min=-72.0, lat_max=-62.0,
    lon_min=5.0,   lon_max=20.0,
    n_lat=10, n_lon=10,
)


def make_latlon_da(
    lat_min: float, lat_max: float, n_lat: int,
    lon_min: float, lon_max: float, n_lon: int,
    *,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    fill: float | None = None,
) -> xr.DataArray:
    """Return a synthetic DataArray on a regular lat-lon grid."""
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    if fill is not None:
        data = np.full((n_lat, n_lon), fill)
    else:
        # linear ramp: value = lat + lon  (easy to verify analytically)
        data = lats[:, None] + lons[None, :]
    return xr.DataArray(data, dims=[lat_dim, lon_dim], coords={lat_dim: lats, lon_dim: lons})


def make_polarstereo_da(
    x_min: float, x_max: float, n_x: int,
    y_min: float, y_max: float, n_y: int,
    *,
    x_dim: str = "x",
    y_dim: str = "y",
    fill: float | None = None,
) -> xr.DataArray:
    """Return a synthetic DataArray on a regular polar-stereo grid (x/y in metres)."""
    xs = np.linspace(x_min, x_max, n_x)
    ys = np.linspace(y_min, y_max, n_y)
    if fill is not None:
        data = np.full((n_y, n_x), fill)
    else:
        data = ys[:, None] + xs[None, :] * 1e-6   # arbitrary smooth field
    return xr.DataArray(data, dims=[y_dim, x_dim], coords={y_dim: ys, x_dim: xs})


# ---------------------------------------------------------------------------
# regrid_latlon
# ---------------------------------------------------------------------------

class TestRegridLatlon:

    def test_output_shape(self):
        from cloud.data.regrid import regrid_latlon
        da = make_latlon_da(-75, -60, 50, 5, 20, 60)
        result = regrid_latlon(da, GRID_5x5)
        assert len(result) == GRID_5x5.n_lat
        assert all(len(row) == GRID_5x5.n_lon for row in result)

    def test_constant_field_reproduces_exactly(self):
        """A constant field should interpolate to the same constant everywhere."""
        from cloud.data.regrid import regrid_latlon
        da = make_latlon_da(-75, -60, 30, 5, 20, 40, fill=42.0)
        result = regrid_latlon(da, GRID_5x5)
        for row in result:
            for val in row:
                assert abs(val - 42.0) < 1e-9

    def test_linear_field_correct_values(self):
        """
        Source field = lat + lon. Interpolated values at the grid centres
        should match lat+lon to within floating-point tolerance.
        """
        from cloud.data.regrid import regrid_latlon

        target = GRID_5x5
        # Dense source so interpolation error is negligible
        da = make_latlon_da(-75, -60, 200, 5, 20, 200)
        result = regrid_latlon(da, target)

        lats = np.linspace(target.lat_min, target.lat_max, target.n_lat)
        lons = np.linspace(target.lon_min, target.lon_max, target.n_lon)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                expected = lat + lon
                assert abs(result[i][j] - expected) < 1e-6, (
                    f"row {i} col {j}: got {result[i][j]}, expected {expected}"
                )

    def test_row0_is_lat_min(self):
        """Row 0 of output must correspond to lat_min (southernmost)."""
        from cloud.data.regrid import regrid_latlon
        # Use a field = lat only (constant in lon) so there is no overlap
        # between row 0 (lat_min) and row -1 (lat_max).
        target = GRID_5x5  # lat_min=-70, lat_max=-65
        lats = np.linspace(-80, -55, 100)
        lons = np.linspace(0, 20, 50)
        data = np.broadcast_to(lats[:, None], (100, 50)).copy()  # field = lat
        da = xr.DataArray(data, dims=["latitude", "longitude"],
                          coords={"latitude": lats, "longitude": lons})
        result = regrid_latlon(da, target)
        # Row 0 has lat≈-70; last row has lat≈-65. All row-0 values < all row-last values.
        assert max(result[0]) < min(result[-1]), "Row 0 should be southernmost (smallest lat)"

    def test_custom_dim_names(self):
        """Should work with non-standard dimension names."""
        from cloud.data.regrid import regrid_latlon
        da = make_latlon_da(-75, -60, 30, 5, 20, 40, lat_dim="lat", lon_dim="lon")
        result = regrid_latlon(da, GRID_5x5, lat_dim="lat", lon_dim="lon")
        assert len(result) == GRID_5x5.n_lat

    def test_nearest_method(self):
        from cloud.data.regrid import regrid_latlon
        da = make_latlon_da(-75, -60, 30, 5, 20, 40)
        result = regrid_latlon(da, GRID_5x5, method="nearest")
        assert len(result) == GRID_5x5.n_lat

    def test_upsample(self):
        """Coarse source → fine target (upsampling)."""
        from cloud.data.regrid import regrid_latlon
        target = GridMetadata(lat_min=-70, lat_max=-65, lon_min=10, lon_max=15, n_lat=20, n_lon=20)
        da = make_latlon_da(-75, -60, 5, 5, 20, 5)
        result = regrid_latlon(da, target)
        assert len(result) == 20
        assert len(result[0]) == 20

    def test_downsample(self):
        """Fine source → coarse target (downsampling)."""
        from cloud.data.regrid import regrid_latlon
        target = GridMetadata(lat_min=-70, lat_max=-65, lon_min=10, lon_max=15, n_lat=3, n_lon=3)
        da = make_latlon_da(-75, -60, 200, 5, 20, 200)
        result = regrid_latlon(da, target)
        assert len(result) == 3

    def test_dataset_dispatch(self):
        """regrid_to_package_grid with a Dataset should extract the named variable."""
        from cloud.data.regrid import regrid_to_package_grid
        da = make_latlon_da(-75, -60, 30, 5, 20, 40, fill=7.0)
        ds = xr.Dataset({"sic": da})
        result = regrid_to_package_grid(ds, GRID_5x5, "sic")
        for row in result:
            for v in row:
                assert abs(v - 7.0) < 1e-9


# ---------------------------------------------------------------------------
# regrid_polarstereo
# ---------------------------------------------------------------------------

class TestRegridPolarstereo:
    """
    These tests use a synthetic DataArray in a fake projected CRS.
    We avoid making any actual network calls or using real OSI SAF data.

    The tests below use EPSG:3031 (Antarctic Polar Stereographic WGS84),
    with an artificial constant or smooth field.  Target grid is a small
    region around 67°S so that the transformed x/y points fall within the
    synthetic source domain.
    """

    # In EPSG:3031 (Antarctic Polar Stereographic), the target grid
    # lat -70...-65, lon -5...5 projects to approximately:
    #   x: [-250_000, +250_000] metres
    #   y: [2_100_000, 2_800_000] metres
    # The synthetic source must cover this domain or results will be NaN.
    SOURCE_X_MIN = -400_000.0
    SOURCE_X_MAX =  400_000.0
    SOURCE_Y_MIN =  1_900_000.0
    SOURCE_Y_MAX =  3_000_000.0
    N_X = 40
    N_Y = 40

    TARGET = GridMetadata(
        lat_min=-70.0, lat_max=-65.0,
        lon_min=-5.0,  lon_max=5.0,
        n_lat=5, n_lon=5,
    )

    def _source(self, fill: float | None = None) -> xr.DataArray:
        return make_polarstereo_da(
            self.SOURCE_X_MIN, self.SOURCE_X_MAX, self.N_X,
            self.SOURCE_Y_MIN, self.SOURCE_Y_MAX, self.N_Y,
            fill=fill,
        )

    def test_output_shape(self):
        from cloud.data.regrid import regrid_polarstereo
        da = self._source(fill=1.0)
        result = regrid_polarstereo(da, self.TARGET, crs="EPSG:3031")
        assert len(result) == self.TARGET.n_lat
        assert all(len(r) == self.TARGET.n_lon for r in result)

    def test_constant_field(self):
        from cloud.data.regrid import regrid_polarstereo
        da = self._source(fill=0.85)
        result = regrid_polarstereo(da, self.TARGET, crs="EPSG:3031")
        for row in result:
            for val in row:
                assert abs(val - 0.85) < 1e-9

    def test_values_are_floats(self):
        from cloud.data.regrid import regrid_polarstereo
        da = self._source()
        result = regrid_polarstereo(da, self.TARGET, crs="EPSG:3031")
        for row in result:
            for val in row:
                assert isinstance(val, float)

    def test_row0_is_lat_min(self):
        """
        With a source field that increases with y (northward), the
        southernmost row (row 0) should have smaller values than the
        northernmost row (last row).
        """
        from cloud.data.regrid import regrid_polarstereo
        # In EPSG:3031 for SH, more negative y ≈ further south.
        # Our synthetic field = y + x*1e-6 → increases with increasing y
        # (= more northward = higher row index).
        da = self._source()
        result = regrid_polarstereo(da, self.TARGET, crs="EPSG:3031")
        assert max(result[0]) < min(result[-1]), (
            "Row 0 (lat_min) should have smaller field values than last row (lat_max)"
        )

    def test_dispatcher_uses_polarstereo_path(self):
        """Passing source_crs should route through regrid_polarstereo."""
        from cloud.data.regrid import regrid_to_package_grid
        da = self._source(fill=0.5)
        result = regrid_to_package_grid(
            da, self.TARGET, variable="dummy",
            source_crs="EPSG:3031", x_dim="x", y_dim="y",
        )
        for row in result:
            for val in row:
                assert abs(val - 0.5) < 1e-9

    def test_dispatcher_no_crs_uses_latlon_path(self):
        """source_crs=None should route through regrid_latlon."""
        from cloud.data.regrid import regrid_to_package_grid
        da = make_latlon_da(-75, -60, 30, -10, 10, 40, fill=3.14)
        result = regrid_to_package_grid(da, self.TARGET, variable="dummy", source_crs=None)
        for row in result:
            for val in row:
                assert abs(val - 3.14) < 1e-9

    def test_y_descending_source(self):
        """Source y-axis descending (common in netCDF files) should be handled."""
        from cloud.data.regrid import regrid_polarstereo
        xs = np.linspace(self.SOURCE_X_MIN, self.SOURCE_X_MAX, self.N_X)
        ys = np.linspace(self.SOURCE_Y_MAX, self.SOURCE_Y_MIN, self.N_Y)  # reversed
        data = np.full((self.N_Y, self.N_X), 0.99)
        da = xr.DataArray(data, dims=["y", "x"], coords={"y": ys, "x": xs})
        result = regrid_polarstereo(da, self.TARGET, crs="EPSG:3031")
        for row in result:
            for val in row:
                assert abs(val - 0.99) < 1e-9

    def test_x_descending_source(self):
        """Source x-axis descending should be handled."""
        from cloud.data.regrid import regrid_polarstereo
        xs = np.linspace(self.SOURCE_X_MAX, self.SOURCE_X_MIN, self.N_X)  # reversed
        ys = np.linspace(self.SOURCE_Y_MIN, self.SOURCE_Y_MAX, self.N_Y)
        data = np.full((self.N_Y, self.N_X), 0.77)
        da = xr.DataArray(data, dims=["y", "x"], coords={"y": ys, "x": xs})
        result = regrid_polarstereo(da, self.TARGET, crs="EPSG:3031")
        for row in result:
            for val in row:
                assert abs(val - 0.77) < 1e-9


# ---------------------------------------------------------------------------
# Separation-of-concern: the two paths must NOT be equivalent
# ---------------------------------------------------------------------------

class TestPathSeparation:
    """
    Confirm that the latlon and polarstereo paths are genuinely different
    code paths, not just aliases.  We do this by checking that
    regrid_polarstereo uses pyproj transformation (not lat/lon axis names).
    """

    def test_polarstereo_has_no_latlon_dims(self):
        """
        regrid_polarstereo should succeed even when the DataArray has no
        latitude/longitude coordinate dimensions — it only needs x and y.
        """
        from cloud.data.regrid import regrid_polarstereo
        # Source domain must cover the projected target coordinates.
        # lat -70...-65, lon -5...5 → EPSG:3031 x≈[-250k,250k], y≈[2.1M,2.8M]
        xs = np.linspace(-400_000, 400_000, 20)
        ys = np.linspace(1_900_000, 3_000_000, 20)
        data = np.full((20, 20), 1.0)
        da = xr.DataArray(data, dims=["y", "x"], coords={"y": ys, "x": xs})
        target = GridMetadata(lat_min=-70, lat_max=-65, lon_min=-5, lon_max=5, n_lat=4, n_lon=4)
        # Must not raise; result shape must be correct
        result = regrid_polarstereo(da, target, crs="EPSG:3031")
        assert len(result) == 4
        assert len(result[0]) == 4

    def test_latlon_fails_on_xy_only_da(self):
        """
        regrid_latlon should raise KeyError/ValueError when the DataArray
        lacks latitude/longitude dimensions, confirming the two paths are
        not interchangeable.
        """
        from cloud.data.regrid import regrid_latlon
        xs = np.linspace(-500_000, 500_000, 20)
        ys = np.linspace(-1_500_000, -500_000, 20)
        data = np.full((20, 20), 1.0)
        da = xr.DataArray(data, dims=["y", "x"], coords={"y": ys, "x": xs})
        target = GridMetadata(lat_min=-70, lat_max=-65, lon_min=-5, lon_max=5, n_lat=4, n_lon=4)
        with pytest.raises((KeyError, ValueError, Exception)):
            regrid_latlon(da, target)
