"""
Forecast endpoint.

Phase 1: returns a mock ForecastPackage generated from the request.
Phase 3+: will call real sea-ice and iceberg ML inference, fetch
          weather/ocean from external providers, assemble and return
          the package.
"""

from fastapi import APIRouter

from data.mock.generator import generate_mock_forecast_package
from shared.schemas.forecast import ForecastPackage
from shared.schemas.vessel import ForecastRequest

router = APIRouter()


@router.post("/forecast", response_model=ForecastPackage)
async def get_forecast(request: ForecastRequest) -> ForecastPackage:
    """
    Return a ForecastPackage for the vessel's area and horizons.

    Currently backed by the mock generator. When ML models and real data
    connectors are ready, this handler is replaced — the request/response
    schema stays the same.
    """
    # request.route_area is accepted by ForecastRequest (open dict for
    # bounding-box / corridor / future formats) but is not consumed in Phase 1.
    # The mock generator covers a fixed ±10° area around the vessel position.
    # Real corridor-constrained packages will be assembled in Phase 3+.
    return generate_mock_forecast_package(
        vessel_position=request.vessel_position,
        destination=request.destination,
        forecast_horizons_hours=request.forecast_horizons_hours,
    )
