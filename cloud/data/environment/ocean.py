"""Stable environment-layer wrapper for the CMEMS ocean connector.

The provider-specific implementation lives in ``cloud.data.ocean.cmems``.
This module is the environment-layer API so callers do not need to know the
provider implementation path.
"""
from __future__ import annotations

from datetime import datetime

from shared.schemas.common import GridMetadata
from shared.schemas.forecast import OceanSnapshot


def fetch_ocean_snapshot(
    grid: GridMetadata,
    valid_at: datetime | None = None,
) -> OceanSnapshot:
    """Fetch a spatially representative CMEMS ocean snapshot for ``grid``.

    ``valid_at`` is accepted for API stability. The current CMEMS connector
    retrieves a daily analysis/mean for a requested calendar date, so the
    date component is forwarded when supplied.
    """
    from cloud.data.ocean.cmems import fetch_ocean_snapshot as fetch_cmems

    target_date = valid_at.date() if valid_at is not None else None
    return fetch_cmems(
        lat_min=grid.lat_min,
        lat_max=grid.lat_max,
        lon_min=grid.lon_min,
        lon_max=grid.lon_max,
        target_date=target_date,
    )
