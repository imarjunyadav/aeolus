# Aeolus Phase 3 data sources

## International sources

- NSIDC G02135 — Antarctic/Southern Hemisphere sea-ice concentration baseline.
- OSI SAF — near-real-time sea-ice concentration and uncertainty.
- USNIC — Antarctic iceberg positions.
- CMEMS — ocean currents/SST and related ocean state.
- ECMWF IFS Open Data — atmospheric and wave forecast.

## Indian government sources

### NPDC / NCPOR

The Indian National Polar Data Center (NPDC) catalogs Indian polar observations.
For Antarctica, the portal lists AWS and synoptic/weather observations from
Maitri and Bharati, including temperature, humidity, pressure and wind fields.
Some datasets are supplied through request/search/download workflows rather than
a stable public API. Aeolus therefore supports an exported CSV/JSON feed or a
configured HTTPS export through `AEOLUS_NPDC_EXPORT_URL`.

The adapter keeps NPDC observations as **observations/provenance**, not as a
forecast. When configured, the closest-in-time observation is attached to the
ECMWF weather snapshot metadata for independent validation/context.

### NCMRWF

NCMRWF's coupled atmosphere/ocean/sea-ice system includes Antarctic sea-ice
fraction and drift products. The source is registered in Aeolus, but a live
connector is not fabricated until an approved machine-readable download/API
route is configured. This avoids hard-coding an undocumented endpoint.

## Phase-3 baseline rule

Until ML forecasting is implemented, real-mode sea ice and iceberg trajectories
use an explicit **persistence baseline**:

- sea-ice concentration is carried forward from the latest retrieved field;
- uncertainty grows with forecast horizon and source age;
- iceberg position is carried forward and its uncertainty radius grows with
  horizon.

These are baseline estimates, **not ML predictions**. They exist so the rest of
Aeolus can be exercised end-to-end and later replaced by trained models without
changing the ForecastPackage contract.
