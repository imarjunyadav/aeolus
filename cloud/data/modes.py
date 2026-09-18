"""
Data mode control for the cloud component.

Set AEOLUS_DATA_MODE=real in the environment to activate real data connectors.
The default is "mock" so development and tests work without credentials.

In REAL mode, connector failures raise ConnectorError — they are never
silently replaced with mock data. The API layer converts ConnectorError
to an appropriate HTTP error response.
"""

from __future__ import annotations

import os
from enum import Enum


class DataMode(str, Enum):
    MOCK = "mock"
    REAL = "real"


def get_data_mode() -> DataMode:
    """Return the active DataMode from the AEOLUS_DATA_MODE env var."""
    raw = os.environ.get("AEOLUS_DATA_MODE", "mock").strip().lower()
    try:
        return DataMode(raw)
    except ValueError:
        raise ValueError(
            f"AEOLUS_DATA_MODE={raw!r} is not valid. Use 'mock' or 'real'."
        )
