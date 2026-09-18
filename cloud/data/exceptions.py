"""
Connector exceptions for real-data mode.

In REAL mode, any failure in a data connector raises one of these.
They are never swallowed silently — the caller (assembler or API route)
decides whether to propagate as HTTP 502 or surface as a degraded status.
"""


class ConnectorError(RuntimeError):
    """Base class for all data connector failures."""


class SeaIceConnectorError(ConnectorError):
    """Raised when the sea-ice connector (OSI SAF or NSIDC) fails."""


class IcebergConnectorError(ConnectorError):
    """Raised when the iceberg connector (USNIC) fails."""


class WeatherConnectorError(ConnectorError):
    """Raised when the weather connector (ECMWF or GFS) fails."""


class OceanConnectorError(ConnectorError):
    """Raised when the ocean connector (CMEMS) fails."""
