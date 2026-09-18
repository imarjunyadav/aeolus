"""Cloud forecast endpoint for mock and real Phase-3 assembly."""

from fastapi import APIRouter, HTTPException

from cloud.data.assembler import build_forecast_package
from cloud.data.exceptions import ConnectorError
from shared.schemas.forecast import ForecastPackage
from shared.schemas.vessel import ForecastRequest

router = APIRouter()


@router.post("/forecast", response_model=ForecastPackage)
async def get_forecast(request: ForecastRequest) -> ForecastPackage:
    try:
        return build_forecast_package(request)
    except ConnectorError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
