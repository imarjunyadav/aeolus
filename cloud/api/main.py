from fastapi import FastAPI

from cloud.api.routes.forecast import router as forecast_router

app = FastAPI(
    title="Aeolus Cloud API",
    description="Environmental forecast service for the Aeolus Antarctic Navigation DSS",
    version="0.1.0",
)

app.include_router(forecast_router, tags=["forecast"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "aeolus-cloud-api"}
