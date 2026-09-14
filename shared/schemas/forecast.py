from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import GridMetadata, Position


class SeaIceForecastHorizon(BaseModel):
    horizon_hours: int
    valid_at: datetime
    # 2-D arrays stored as nested Python lists: shape [n_lat][n_lon], values 0.0–1.0.
    # Using lists keeps the schema JSON-native without binary encoding.
    # Format can be upgraded (e.g. compressed binary) later without changing the outer schema.
    concentration_grid: list[list[float]]
    uncertainty_grid: list[list[float]]


class SeaIceForecast(BaseModel):
    source_data_timestamp: datetime
    source_data_age_seconds: int
    grid_metadata: GridMetadata
    horizons: list[SeaIceForecastHorizon]
    # Extensible: connectors can add provider-specific quality flags here.
    quality_flags: dict[str, str] = Field(default_factory=dict)


class IcebergTrajectoryHorizon(BaseModel):
    horizon_hours: int
    valid_at: datetime
    latitude: float
    longitude: float
    uncertainty_radius_km: float


class IcebergForecast(BaseModel):
    iceberg_id: str
    last_known_position: Position
    last_known_timestamp: datetime
    source_data_age_seconds: int
    # Populated only when the source provides it; not all providers include size.
    size_category: Optional[str] = None
    horizons: list[IcebergTrajectoryHorizon]


class WeatherSnapshot(BaseModel):
    """
    Area-representative atmospheric values for the forecast corridor.

    Phase 1 placeholder: a single snapshot covers the whole forecast area.
    Future phases will supply corridor-resolved and time-horizon-resolved
    values once provider evaluation (Phase 2) determines the best sources.
    Gridded weather arrays can be added as Optional[list[list[float]]] fields
    in a later schema version without breaking existing clients.

    These are consumed inputs from external providers; Aeolus does not
    produce weather forecasts. All fields are Optional because not every
    provider or operational mode populates every variable.
    """
    source_data_timestamp: datetime
    source_data_age_seconds: int
    valid_at: datetime
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    pressure_hpa: Optional[float] = None
    significant_wave_height_m: Optional[float] = None
    air_temperature_c: Optional[float] = None
    # Lists which of the above Optional fields are actually populated.
    available_variables: list[str] = Field(default_factory=list)
    # Extensible: provider-specific metadata (e.g. model cycle time).
    metadata: dict[str, Any] = Field(default_factory=dict)


class OceanSnapshot(BaseModel):
    """
    Area-representative oceanographic values for the forecast corridor.

    Phase 1 placeholder: a single snapshot covers the whole forecast area.
    Future phases will supply corridor-resolved and time-horizon-resolved
    values once provider evaluation (Phase 2) determines the best sources.
    Gridded arrays can be added as Optional fields in later schema versions.

    Like WeatherSnapshot, these are consumed inputs, not Aeolus predictions.
    """
    source_data_timestamp: datetime
    source_data_age_seconds: int
    valid_at: datetime
    current_u_ms: Optional[float] = None   # eastward component
    current_v_ms: Optional[float] = None   # northward component
    sst_c: Optional[float] = None
    available_variables: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastPackage(BaseModel):
    """
    The complete environmental package produced by the cloud and consumed by
    the vessel. This is the primary interface between the two components.

    The vessel stores this in its local cache and uses it for all navigation
    calculations until a fresher package is received.
    """
    package_id: str
    schema_version: str = "1.0"
    generated_at: datetime
    generated_by: str             # cloud API identifier, e.g. "aeolus-cloud-api/0.1"
    valid_from: datetime
    valid_to: datetime
    # Echo of the area the package covers; structure is intentionally open so
    # the cloud can evolve bounding-box / corridor / point representations.
    request_area: Optional[dict[str, Any]] = None
    sea_ice: SeaIceForecast
    icebergs: list[IcebergForecast]
    weather: WeatherSnapshot
    ocean: OceanSnapshot
    # Confidence at time of generation. The vessel tracks its own freshness
    # independently in ForecastFreshness; this value is the cloud's assessment.
    overall_confidence: float = Field(ge=0.0, le=1.0)
