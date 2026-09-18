"""Small durable JSON cache for the onboard ForecastPackage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.schemas.forecast import ForecastPackage


class ForecastCache:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or "vessel/cache/forecast_package.json")

    def save(self, package: ForecastPackage) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(package.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(self.path)

    def load(self) -> ForecastPackage | None:
        if not self.path.exists():
            return None
        return ForecastPackage.model_validate(json.loads(self.path.read_text(encoding="utf-8")))

    def exists(self) -> bool:
        return self.path.exists()
