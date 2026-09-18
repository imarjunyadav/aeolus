"""Onboard API: durable forecast sync plus deterministic navigation MVP."""

import asyncio
import json
import os
from datetime import datetime, timezone
from urllib import request as http_request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.schemas.forecast import ForecastPackage
from shared.schemas.navigation import NavigationResult
from shared.schemas.vessel import ForecastRequest, VesselState
from vessel.cache.store import ForecastCache
from vessel.navigation.planner import plan_navigation

app = FastAPI(
    title="Aeolus Vessel API",
    description="Local decision-support service running on the research vessel",
    version="0.2.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def _cache() -> ForecastCache:
    return ForecastCache(os.getenv("AEOLUS_FORECAST_CACHE", "vessel/cache/forecast_package.json"))


@app.get("/health")
async def health() -> dict:
    package = _cache().load()
    return {"status": "ok", "service": "aeolus-vessel-api", "forecast_cached": package is not None, "forecast_package_id": package.package_id if package else None}


def _fetch_cloud_forecast(cloud_url: str, payload: dict) -> ForecastPackage:
    request = http_request.Request(
        f"{cloud_url.rstrip('/')}/forecast",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "aeolus-vessel/0.2"},
        method="POST",
    )
    with http_request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
        if response.status >= 400:
            raise RuntimeError(body)
    return ForecastPackage.model_validate(json.loads(body))


@app.post("/sync")
async def sync_forecast(request: ForecastRequest | None = None) -> dict:
    # Preserve the Phase-1 health/compatibility probe: an empty POST is not a sync request.
    if request is None:
        return {"status": "not_implemented", "phase": 5, "hint": "POST a ForecastRequest to perform a real sync."}
    cloud_url = os.getenv("CLOUD_API_URL", "http://localhost:8000")
    try:
        package = await asyncio.to_thread(_fetch_cloud_forecast, cloud_url, request.model_dump(mode="json"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Cloud synchronization failed: {exc}") from exc
    _cache().save(package)
    return {"status": "synced", "package_id": package.package_id, "received_at": datetime.now(tz=timezone.utc).isoformat()}


@app.post("/navigate", response_model=NavigationResult | dict)
async def navigate(state: VesselState | None = None):
    # Preserve the Phase-1 compatibility probe for an empty POST.
    if state is None:
        return {"status": "not_implemented", "phase": 6, "hint": "POST a VesselState after /sync."}
    package = _cache().load()
    if package is None:
        raise HTTPException(status_code=409, detail="No ForecastPackage is cached. Call /sync first.")
    try:
        return plan_navigation(package, state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
