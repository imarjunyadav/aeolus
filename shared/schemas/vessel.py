from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import Position


class VesselContext(BaseModel):
    """
    Vessel capability and constraint parameters.

    These are static or semi-static properties used by Safety Shield and
    Navigation Intelligence. Add fields here as constraints are identified
    from actual vessel data.
    """
    vessel_id: str
    ice_class: Optional[str] = None                  # e.g. "PC5", "1A", "ICE-C"
    max_ice_concentration: Optional[float] = None    # hard limit: 0.0–1.0
    min_iceberg_clearance_km: Optional[float] = None
    # Extensible: additional capability/constraint fields added without restructuring.
    extra: dict[str, Any] = Field(default_factory=dict)


class VesselState(BaseModel):
    """Current navigational state of the vessel."""
    vessel_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    speed_knots: float
    heading_deg: float
    destination: Optional[Position] = None
    context: Optional[VesselContext] = None


class ForecastRequest(BaseModel):
    """
    Body for POST /forecast.

    All fields beyond vessel_position are optional; the cloud API applies
    sensible defaults for missing fields. This structure is designed to grow
    (corridor, multi-destination, custom horizons) without breaking clients
    that only provide vessel_position.
    """
    vessel_position: Position
    destination: Optional[Position] = None
    # Open dict allows bounding-box, polygon, corridor, or other area
    # representations to be added without a schema change.
    route_area: Optional[dict[str, Any]] = None
    forecast_horizons_hours: list[int] = Field(
        default=[6, 12, 24, 48, 72],
        description="Requested forecast horizons in hours",
    )
    vessel_context: Optional[VesselContext] = None
