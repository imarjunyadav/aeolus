"""
Data connectors and preprocessing pipeline.

Phase 3: real data connectors for sea ice, icebergs, weather, and ocean.

Mode is controlled by the AEOLUS_DATA_MODE environment variable:
  mock (default) — synthetic data from the mock generator; no credentials needed
  real           — real data from external providers; ConnectorError on failure
"""

from .exceptions import (
    ConnectorError,
    IcebergConnectorError,
    OceanConnectorError,
    SeaIceConnectorError,
    WeatherConnectorError,
)
from .modes import DataMode, get_data_mode

__all__ = [
    "ConnectorError",
    "DataMode",
    "IcebergConnectorError",
    "OceanConnectorError",
    "SeaIceConnectorError",
    "WeatherConnectorError",
    "get_data_mode",
]
