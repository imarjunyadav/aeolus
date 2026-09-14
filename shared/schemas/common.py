from enum import Enum

from pydantic import BaseModel


class ConnectivityState(str, Enum):
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class SafetyStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class WarnLevel(str, Enum):
    INFO = "INFO"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    DANGER = "DANGER"


class WarnCategory(str, Enum):
    ICEBERG_ENCOUNTER = "ICEBERG_ENCOUNTER"
    ICE_CONCENTRATION = "ICE_CONCENTRATION"
    WEATHER = "WEATHER"
    DATA_STALE = "DATA_STALE"
    NO_SAFE_ROUTE = "NO_SAFE_ROUTE"
    RESTRICTED_AREA = "RESTRICTED_AREA"


class OptimizationMode(str, Enum):
    BALANCED = "BALANCED"
    FUEL = "FUEL"
    RISK = "RISK"
    SHORTEST = "SHORTEST"


class Position(BaseModel):
    latitude: float
    longitude: float


class GridMetadata(BaseModel):
    """Defines the lat/lon grid that sea-ice forecast arrays map onto."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n_lat: int
    n_lon: int

    @property
    def lat_resolution_deg(self) -> float:
        return (self.lat_max - self.lat_min) / max(self.n_lat - 1, 1)

    @property
    def lon_resolution_deg(self) -> float:
        return (self.lon_max - self.lon_min) / max(self.n_lon - 1, 1)
