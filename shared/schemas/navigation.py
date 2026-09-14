from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .common import (
    ConnectivityState,
    OptimizationMode,
    Position,
    RiskLevel,
    SafetyStatus,
    WarnCategory,
    WarnLevel,
)


class RouteWaypoint(BaseModel):
    position: Position
    estimated_arrival: Optional[datetime] = None
    ice_concentration_at_point: Optional[float] = None
    risk_score_at_point: Optional[float] = None


class RouteCandidate(BaseModel):
    """
    A single candidate route produced by the route optimizer.

    Safety Shield inspects each candidate and sets safety_status.
    Only APPROVED routes are eligible for ranking and recommendation.
    """
    route_id: str
    optimization_mode: OptimizationMode
    waypoints: list[RouteWaypoint]
    total_distance_nm: float
    # Fuel is expressed relative to a baseline for MVP; absolute units depend
    # on vessel-specific fuel consumption models added in a later phase.
    estimated_fuel_relative: Optional[float] = None
    estimated_eta_hours: Optional[float] = None
    risk_score: Optional[float] = None           # 0.0–1.0 aggregate
    risk_level: Optional[RiskLevel] = None
    safety_status: SafetyStatus
    rejection_reasons: list[str] = Field(default_factory=list)


class NavigationWarning(BaseModel):
    """
    An active navigation alert issued by Navigation Intelligence.

    TTE (time_to_encounter_hours) lives here as an informational/alert field.
    It is NOT used directly as a route rejection criterion; rejection is
    determined by predicted encounter conditions, clearance, and defined
    safety constraints evaluated by Safety Shield.
    """
    warning_id: str
    level: WarnLevel
    category: WarnCategory
    message: str
    iceberg_id: Optional[str] = None
    time_to_encounter_hours: Optional[float] = None
    estimated_clearance_km: Optional[float] = None
    issued_at: datetime


class ForecastFreshness(BaseModel):
    """
    Vessel-side tracking of how current the cached forecast data is.

    current_confidence is intentionally left as Optional[float] with no
    decay formula applied yet. The actual confidence decay policy will be
    determined after real dataset inspection and forecast-error analysis.
    The fields are all present so decay logic can be plugged in later
    without changing this schema.
    """
    package_id: str
    package_received_at: datetime
    sea_ice_source_data_age_seconds: int
    iceberg_source_data_age_seconds: int
    weather_source_data_age_seconds: int
    ocean_source_data_age_seconds: int
    # Populated by decay policy when implemented; None means not yet computed.
    current_confidence: Optional[float] = None
    connectivity_state: ConnectivityState


class NavigationResult(BaseModel):
    """
    Complete output of one navigation intelligence cycle.

    recommended_route is None when no_safe_route is True.
    Safety Shield runs before ranking; only APPROVED routes reach ranking.
    """
    result_id: str
    computed_at: datetime
    vessel_state_timestamp: datetime
    forecast_package_id: str
    recommended_route: Optional[RouteCandidate] = None
    alternative_routes: list[RouteCandidate] = Field(default_factory=list)
    rejected_routes: list[RouteCandidate] = Field(default_factory=list)
    no_safe_route: bool = False
    no_safe_route_reason: Optional[str] = None
    active_warnings: list[NavigationWarning] = Field(default_factory=list)
    forecast_freshness: ForecastFreshness
    connectivity_state: ConnectivityState
