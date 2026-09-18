# Aeolus — Placeholder / Replacement Register

This file is intentionally explicit: a green test suite does not mean every
scientific or operational component is production-ready.

## Must be replaced before operational deployment

1. **Sea-ice forecasting**
   - Current REAL mode uses an explicitly labelled persistence baseline.
   - Replace with a validated trained model after historical Antarctic
     datasets are assembled and evaluated.
   - Code: `cloud/data/assembler.py`, `cloud/models/sea_ice_forecaster.py`.

2. **Iceberg trajectory prediction**
   - Current REAL mode extrapolates the last observed velocity.
   - Replace/augment with a validated trajectory model and uncertainty model.
   - Code: `cloud/data/assembler.py`, `cloud/models/iceberg_baseline.py`.

3. **NPDC transport**
   - The NPDC adapter consumes an export/downloaded CSV or JSON.
   - `AEOLUS_NPDC_EXPORT_URL` / `AEOLUS_NPDC_EXPORT_PATH` must point to an
     authorized, maintained NPDC delivery mechanism. No undocumented API is
     hard-coded.
   - Code: `cloud/data/india/npdc.py`.

4. **NCMRWF live ingestion**
   - Source is registered, but no machine-readable endpoint is assumed.
   - Replace the source-registration entry with a verified NCMRWF delivery
     interface before enabling it in REAL mode.
   - Code: `cloud/data/india/sources.py`.

5. **Weather/ocean spatial resolution**
   - `WeatherSnapshot` and `OceanSnapshot` are still area/point summaries.
   - Replace with corridor/time-resolved fields if route-level weather/ocean
     gradients materially affect navigation.
   - Code: `shared/schemas/forecast.py`.

6. **Fuel model**
   - `estimated_fuel_relative` is a relative planning metric, not litres or
     tonnes.
   - Replace with a vessel-specific propulsion/fuel model and calibration.
   - Code: `vessel/navigation/planner.py`.

7. **Forecast confidence calibration**
   - Current confidence/freshness handling is operationally conservative but
     heuristic.
   - Replace the heuristic with source- and horizon-specific error
     calibration using held-out historical forecasts.
   - Code: `vessel/navigation/planner.py`.

8. **Production communications and persistence**
   - Local HTTP and a JSON vessel cache are suitable for development.
   - Replace with authenticated ship/cloud communications, durable storage,
     retries, integrity checks and key management before deployment.

## Development-only placeholders

- `data/mock/generator.py` is synthetic test data. Never use it to claim
  real-world forecast performance.
- UI uses `DEMO-VESSEL` as a development default. Replace with a configured
  vessel identity.
- OpenStreetMap tiles are suitable for the MVP UI; establish the permitted
  operational chart/tile source before maritime deployment.
- The linear-trend and constant-velocity models are benchmark baselines, not
  final scientific models.

## No longer a placeholder

- CMEMS provider implementation exists in `cloud/data/ocean/cmems.py`.
- The environment-layer `cloud/data/environment/ocean.py` now delegates to it.
- Phase-3 real assembly exists and is tested.
- Navigation MVP and Safety Shield constraints are implemented and tested.
