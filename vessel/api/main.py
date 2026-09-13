"""
Vessel-side API.

Phase 1: health check and placeholder endpoints only.
Phase 5+: forecast cache sync, navigation result endpoint, vessel state input.

All navigation logic lives in vessel/navigation/ and vessel/safety/.
This API is a thin HTTP interface over those components.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Aeolus Vessel API",
    description="Local decision-support service running on the research vessel",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "aeolus-vessel-api"}


# ---------------------------------------------------------------------------
# Placeholder endpoints — implemented in later phases
# ---------------------------------------------------------------------------

@app.post("/sync")
async def sync_forecast() -> dict:
    """
    Phase 5: triggers a pull from the cloud API, stores result in local cache.
    """
    return {"status": "not_implemented", "phase": 5}


@app.post("/navigate")
async def navigate() -> dict:
    """
    Phase 6+: accepts VesselState, runs navigation intelligence, returns
    NavigationResult.
    """
    return {"status": "not_implemented", "phase": 6}
