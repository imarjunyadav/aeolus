from datetime import datetime, timezone

import numpy as np
import xarray as xr

from cloud.data.modes import DataMode
from cloud.data import assembler
from shared.schemas.common import Position
from shared.schemas.forecast import OceanSnapshot, WeatherSnapshot
from shared.schemas.vessel import ForecastRequest


def test_real_assembler_wires_phase3_sources(monkeypatch):
    now = datetime(2026, 9, 17, tzinfo=timezone.utc)
    grid = assembler.GridMetadata(lat_min=-75, lat_max=-65, lon_min=0, lon_max=10, n_lat=4, n_lon=4)
    da = xr.DataArray(np.full((4, 4), 0.42), dims=("y", "x"), coords={"x": np.arange(4), "y": np.arange(4)}, attrs={"crs": "EPSG:3031"})
    monkeypatch.setattr("cloud.data.sea_ice.nsidc.fetch_concentration_geotiff", lambda target_date: da)
    monkeypatch.setattr("cloud.data.regrid.regrid_to_package_grid", lambda *args, **kwargs: [[0.42] * grid.n_lon for _ in range(grid.n_lat)])
    monkeypatch.setattr("cloud.data.icebergs.usnic.fetch_current_positions", lambda: ([], now))
    monkeypatch.setattr("cloud.data.environment.weather.fetch_weather_forecast", lambda lat, lon, valid_at: WeatherSnapshot(source_data_timestamp=now, source_data_age_seconds=0, valid_at=valid_at, wind_speed_ms=10, available_variables=["wind_speed_ms"]))
    monkeypatch.setattr("cloud.data.ocean.cmems.fetch_ocean_snapshot", lambda **kwargs: OceanSnapshot(source_data_timestamp=now, source_data_age_seconds=0, valid_at=now, current_u_ms=0.1, current_v_ms=0.2, sst_c=-1.0, available_variables=["current_u_ms", "current_v_ms", "sst_c"]))

    req = ForecastRequest(vessel_position=Position(latitude=-70, longitude=5), forecast_horizons_hours=[6, 12])
    package = assembler.build_forecast_package(req, DataMode.REAL)
    assert package.generated_by == "aeolus-real-assembler/1.1"
    assert package.sea_ice.quality_flags["forecast_method"] == "persistence_baseline"
    assert len(package.sea_ice.horizons) == 2
    assert package.sea_ice.horizons[0].concentration_grid[0][0] == 0.42
