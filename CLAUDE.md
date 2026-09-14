# Aeolus — Antarctic Navigation Decision Support System

AI-enabled, cloud-assisted maritime DSS for Antarctic sea-ice and iceberg
forecasting, risk assessment, and route recommendation.

## Project context

SIH 2026 Problem Statement 26059 (MoES / NCPOR).
Full requirements and architecture: see master context document.

## Repository structure

```
aeolus/
├── shared/schemas/      Core data schemas (Pydantic). Single source of truth
│                        for all inter-component contracts.
├── cloud/
│   ├── data/            Data connectors and preprocessing (Phase 2)
│   ├── models/          ML training and inference (Phase 3)
│   └── api/             FastAPI: POST /forecast → ForecastPackage
├── vessel/
│   ├── cache/           Local forecast cache, SQLite (Phase 5)
│   ├── navigation/      Risk, TTE, fuel, ETA, routing (Phase 6)
│   ├── safety/          Safety Shield + No Safe Route (Phase 7)
│   └── api/             Local FastAPI for UI consumption
├── ui/                  React + Leaflet decision support UI (Phase 9)
├── benchmark/           Historical voyage replay pipeline (Phase 11)
└── data/mock/           Minimal synthetic data generator (integration only)
```

## Key design rules

1. **ML predicts** sea ice and iceberg trajectories only.
2. **Algorithms calculate** risk, TTE, fuel, ETA, routes.
3. **Safety Shield** is deterministic and runs before route ranking.
4. **No Safe Route** is a valid and expected output.
5. **TTE** lives in NavigationWarning (alert/informational), not in
   route rejection logic.
6. **ForecastFreshness** is tracked on the vessel separately from the
   cloud's `overall_confidence`. Confidence decay policy is TBD.
7. **Vessel operates offline** using cached forecasts when connectivity
   is lost.
8. **Benchmark** enforces strict time-gating: data at timestamp > T is
   never visible during replay at time T.

## Running locally

```bash
# Install all dev dependencies
pip install -r requirements-dev.txt

# Set PYTHONPATH so all modules resolve from repo root
export PYTHONPATH=$(pwd)

# Run cloud API
uvicorn cloud.api.main:app --port 8000 --reload

# Run vessel API (separate terminal)
uvicorn vessel.api.main:app --port 8001 --reload
```

## Docker Compose

```bash
docker compose up --build
```

Services:
- `cloud-api` → http://localhost:8000  (docs: /docs)
- `vessel-api` → http://localhost:8001  (docs: /docs)

## Current phase

**Phase 1 complete** — shared schemas, mock generator, project skeleton,
Docker Compose.

Phase 2: provider/data-source evaluation (next).
Track B (vessel navigation engine) is unblocked and can run against
mock ForecastPackage data.

## ForecastPackage

The primary interface between cloud and vessel (~32 KB / package at
20×20 grid, 5 horizons, 4 icebergs). Vessel pulls via `POST /forecast`.

Key fields: `sea_ice` (grid + uncertainty per horizon), `icebergs`
(trajectory + uncertainty per horizon), `weather` (snapshot),
`ocean` (snapshot), freshness metadata, `overall_confidence`.

## Schema versioning

`ForecastPackage.schema_version` is set to `"1.0"`. Increment minor
version for additive changes; major version for breaking changes.
