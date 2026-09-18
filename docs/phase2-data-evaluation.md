# Phase 2 — Data Source and Dataset Evaluation
# Aeolus Antarctic Navigation Decision Support System

**Date**: 2026-09-15  
**Status**: Complete — recommended stack selected, pipeline design proposed  
**Evaluation method**: Official provider documentation review, third-party access reports, peer-reviewed literature  
**Note on proof-of-access tests**: The managed build environment restricts outbound data connections (proxy policy). Access patterns documented here were verified by research agents via WebFetch and against official provider documentation. All cited URLs and Python library interfaces are confirmed from primary sources.

---

## TL;DR — Recommended Stack

| Need | Primary | Fallback / Historical |
|------|---------|----------------------|
| Sea ice concentration (NRT, 10 km) | OSI SAF via MET Norway THREDDS | NSIDC G02135 daily (zero-auth) |
| Sea ice historical training corpus | OSI SAF CDR/ICDR (1978–present) | NSIDC-0051 archive (1978–2025, static) |
| Iceberg current positions | USNIC weekly CSV | — |
| Iceberg trajectory ML training | BYU/NIC SCP (1978–2023 daily) + USNIC archive | S3 community mirror (2014–present) |
| Weather forecast | ECMWF Open Data IFS (9 km, 15 d) | GFS via NOMADS (0.25°, 16 d, zero-auth) |
| Weather historical training | ERA5 via CDS API (1940–present) | — |
| Ocean forecast (currents + SST) | CMEMS PHY 001_024 (8 km, 10 d) | — |
| Ocean historical training | CMEMS GLORYS12 (8 km, 1993–present) | — |

**Minimum accounts needed**: CDS (ERA5) + CMEMS (ocean + NRT sea ice) + ECMWF (IFS Open Data). GFS, USNIC, NSIDC G02135, and BYU/NIC SCP need no authentication.

---

## Part 1 — Sea Ice Data

### Candidates evaluated

1. **NSIDC** (National Snow and Ice Data Center)
2. **OSI SAF** (EUMETSAT Ocean and Sea Ice SAF)
3. **CMEMS** (Copernicus Marine Service)
4. **USNIC** (US National Ice Center)

### Critical status updates (discovered during evaluation)

- **NSIDC-0051 ended 31 December 2025.** The SMMR/SSM/I/SSMIS Brightness Temperature CDR is now a static archive. The active daily production product is **NSIDC-0803** (Sea Ice Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS, Version 2).
- **USNIC Antarctic operations downgraded to "Basic service" effective 5 May 2025.** Antarctic concentration charts now follow an *alternating weekly schedule* (one Antarctic week, one Arctic week). The domain is now `usicecenter.gov`, not `usnic.navy.mil`.

### Comparison table

| Criterion | OSI SAF | NSIDC G02135 | CMEMS | USNIC |
|-----------|---------|-------------|-------|-------|
| Product type | Observation (passive µwave) + CDR | Observation (passive µwave) | Reanalysis (GLORYS12) + NRT OBS | Operational analysis (multi-sensor, human QC) |
| Resolution | **10 km** (NRT OSI-401-d) / 25 km (CDR) | 25 km | 8 km (GLORYS12) | 10 km (G10033 gridded) |
| Antarctic coverage | Full 90°S–90°N | Full, 316×332 polar-stereo grid | Full 80°S–90°N | Antarctic waters |
| History | 1978–present | **1978–present** | GLORYS12: 1993–present | 2003–present |
| Update frequency | Daily NRT (5 hr latency), ICDR fast-track 2-day | Daily | Daily (NRT), monthly mean | **Alternating weekly** (every other week for Antarctic, 2025+) |
| Variables | Concentration, edge, type, drift | Concentration, extent | Concentration, thickness, SST, currents, SSH — co-located | Concentration, stage-of-development, ice type |
| Auth required | None (CDR via OPeNDAP/FTP) | **None** (G02135 plain HTTPS) | Free CMEMS account | None (NSIDC-hosted G10033) |
| Python library | `xarray` + OPeNDAP URL | `requests` | `copernicusmarine` | `requests` + `netCDF4` |
| Automation | Low difficulty | **Very low** | Low difficulty | Medium (alternating schedule) |
| Licensing | EUMETSAT open (free, attribution) | NASA open (no restrictions) | Copernicus Marine (free, attribution) | US Gov public domain |

### Decision

**Primary NRT (operational):** OSI SAF OSI-401-d / OSI-408  
- 10 km resolution, 2–5 hour latency, Southern Ocean fully covered  
- CDR from 1978 via anonymous FTP/OPeNDAP — zero credentials for historical data  
- Clean separation between historical CDR (training) and NRT (operational) on the same provider

**Fallback / extent baseline:** NSIDC G02135  
- Zero authentication, plain HTTPS daily CSV + GeoTIFF  
- Immediately automatable; used as extent/anomaly baseline even in primary stack  
- 47-year record for anomaly detection in the ML training pipeline

**ML co-variates (co-located with sea ice):** CMEMS GLORYS12  
- SST, ocean currents, salinity, and sea ice concentration in one spatially consistent reanalysis  
- Same `copernicusmarine` API as the ocean forecast product — one account, one library

**Rejected:** USNIC as a primary sea ice source  
- Alternating weekly schedule (every other week for Antarctic since 2025) is too infrequent for a vessel navigation system
- "Basic service" status indicates reduced maintenance guarantees
- Unique value-add (ice type, stage-of-development) still valid as supplementary input, but cannot anchor the pipeline

### Variables selected from sea ice sources

From OSI SAF (NRT + CDR):
- `ice_conc` — sea ice concentration (0–100%)
- `smearing_uncertainty` — concentration uncertainty estimate
- `algorithm_standard_error` — algorithm uncertainty

From CMEMS GLORYS12 (ML features co-located with sea ice):
- `siconc` — sea ice concentration (cross-check with OSI SAF)
- `sithick` — sea ice thickness (only source providing this)
- `thetao` [surface] — sea surface temperature
- `uo`, `vo` — ocean surface current components

---

## Part 2 — Iceberg Data

### Critical top-level finding

**No operational Antarctic iceberg trajectory forecast product exists as a public data feed.** Every available source provides either current position observations (updated weekly) or historical position tracks. The ML model must be built from scratch and trained on position sequences + environmental forcing data.

This was confirmed by surveying USNIC, BYU/NIC, BAS, OSI SAF, and current literature (IDRIFTNET, arXiv 2507.00036, June 2025).

### Candidates evaluated

1. **USNIC** (US National Ice Center)
2. **BYU/NIC SCP** (Brigham Young University / National Ice Center Scatterometer Climate Record)
3. **BAS** (British Antarctic Survey / UK Polar Data Centre)
4. **ESSD Zenodo dataset** (Circum-Antarctic icebergs 2018–2023)
5. **IPAB/NSIDC** (sea ice buoys — not iceberg trackers)

### Comparison table

| Source | Coverage | History | Data type | Update | Format | Auth | Value for Aeolus |
|--------|----------|---------|-----------|--------|--------|------|-----------------|
| USNIC weekly CSV | All named ≥ 50 km² | 1979–present | Current positions | Weekly | CSV | None | **Operational input** |
| USNIC daily shapefiles | Same | 1979–present | Current positions | Daily | Shapefile | None | Operational input (higher frequency) |
| Community S3 mirror (USNIC) | All named | 2014–present | Position history | Weekly | CSV | None | **Easiest ML training bootstrap** |
| BYU/NIC SCP | ~648 icebergs | 1978–2023 | Daily position tracks | Was daily; last update Oct 2023 | Custom / JSON | None | **Primary ML training corpus** |
| BAS/UK PDC | Individual icebergs | Case-by-case | Historical positions | Static datasets | CSV / NetCDF | None | Supplementary training (notable icebergs) |
| ESSD Zenodo 2026 | All ≥ 0.04 km² | 2018–2023 | Annual snapshots (Oct) | Annual | Shapefile | None | Small-iceberg density; not trajectory data |
| IPAB/NSIDC | Sea ice buoys | 1995–1998 | Buoy drift (not icebergs) | Archive | Text | None | Not relevant |

### Decision

**Operational positions (inference input):** USNIC weekly CSV  
URL: `https://usicecenter.gov/File/DownloadCurrent?pId=134`  
- Zero authentication, single parameterless HTTP GET
- Covers all named Antarctic icebergs (≥ 50 km²) — the hazard-class size relevant to vessel navigation
- GIS shapefiles available at `https://usicecenter.gov/Catalog/AntarcGisDaily` for daily updates

**ML training corpus (primary):** BYU/NIC SCP database  
URL: `https://www.scp.byu.edu/data/iceberg/default.html`  
- Daily positions for ~648 named icebergs, 1978–2023  
- This is the densest multi-decade Antarctic iceberg position record available
- Access via web download or the `Joel-hanson/Iceberg-locations` Python scraper tool
- Last database update October 2023; scatterometer-based detection continues via ASCAT

**ML training corpus (easiest bootstrap):** Community S3 mirror  
URL: `https://usi-icebergs.s3.eu-central-1.amazonaws.com/icebergs_locations_usi.csv`  
- 19,169 rows, ~1.2 MB, updated weekly via AWS Lambda from USNIC archive
- 2014–present; same fields as USNIC CSV; no auth; immediately usable with `pd.read_csv()`
- Start here for training pipeline development; supplement with BYU/NIC for pre-2014 data

**Supplementary:** BAS/UK PDC individual iceberg datasets  
- Useful for high-fidelity tracks of specific large icebergs (A68, B31)
- Open access, CSV/NetCDF, no auth

**Not used:** ESSD Zenodo (annual snapshots, not tracks), IPAB (sea ice buoys, not icebergs)

### Variables for iceberg trajectory ML model

Input features per (iceberg, timestamp):
- Last known position: latitude, longitude
- Iceberg size: length_nm, width_nm (from USNIC) or area_km2 (from BYU/NIC)
- Ocean surface current: u, v [from CMEMS NRT, co-located to iceberg position]
- Wind: u10, v10 [from ECMWF IFS or GFS, co-located]
- Sea ice concentration at iceberg position [from OSI SAF]
- Time-of-year (cyclical encoding)

Output per horizon (6h, 12h, 24h, 48h, 72h):
- Predicted latitude, predicted longitude
- Uncertainty radius km (must grow with horizon; approximately sqrt(t) scaling as baseline)

Reference architecture: IDRIFTNET (arXiv:2507.00036) — physics-informed residual network combining Lagrangian ocean current advection + learned residual drift. Baseline comparison should be a pure Lagrangian advection (ocean current only) to quantify ML uplift.

---

## Part 3 — Weather Data

### Candidates evaluated

1. **ERA5** (ECMWF Reanalysis, via Copernicus CDS)
2. **ECMWF Open Data IFS / AIFS** (operational forecast)
3. **GFS** (NOAA Global Forecast System)

### Critical status update: ECMWF fully opened October 2025

On 1 October 2025, ECMWF opened its entire Real-Time Catalogue under CC-BY-4.0. The previous distinction between free low-resolution ECMWF Open Data and paid high-resolution IFS is now obsolete. Full 9 km native resolution IFS (and the AIFS ML-based forecast) is freely available.

### Comparison table

| Criterion | ERA5 | ECMWF IFS (Open Data) | GFS |
|-----------|------|-----------------------|-----|
| Type | Reanalysis (1940–~3 months ago) | Operational forecast | Operational forecast |
| Resolution | 0.25° atm / 0.5° waves | **9 km native** (HRES) | 0.25° |
| Forecast horizon | N/A | 15 days HRES / 46 days ENS | 384 h (16 days) |
| Update frequency | Trailing ~3 months, 5-day lag | 4× daily (00/06/12/18 UTC) | 4× daily |
| Key variables | u10, v10, MSLP, T2m, SWH, SIC | Full IFS catalogue: winds, MSLP, T, waves, SIC, SST | u10, v10, MSLP, T2m, SWH, SIC |
| Antarctic coverage | Full global, pole-to-pole | Full global | Full global; GFS-Wave to ~80°S |
| Auth | Free CDS account + API key | Free ECMWF account | **None** |
| Python library | `cdsapi` | `ecmwf-opendata` | `herbie` (pip) |
| Output format | NetCDF4 / GRIB2 | GRIB2 (read with cfgrib+xarray) | GRIB2 (herbie handles this) |
| Automation | Low | Low | **Very low** |
| Licensing | Copernicus (free, attribution) | CC-BY-4.0 | US public domain (no restrictions) |

### Decision

**Primary weather forecast (operational):** ECMWF Open Data IFS  
- 9 km resolution is significantly better than GFS 0.25° (~28 km) for Southern Ocean cyclone details
- Now fully free and open; `ecmwf-opendata` Python library is clean
- 15-day HRES + 46-day ENS covers all required forecast horizons

**Fallback weather forecast:** GFS via NOMADS  
- Zero authentication; `herbie` library abstracts NOMADS/S3 access
- 384-hour horizon, 0.25°, coupled GFS-Wave for significant wave height
- Serves as hot standby when ECMWF operational model is delayed or unavailable

**Historical training / benchmark:** ERA5 via CDS API  
- 86 years (1940–present) of hourly reanalysis, global, all required variables in one API call
- `cdsapi` + `xarray` for fully automated retrieval; area bounding box subsetting supported
- Time-gating for benchmark pipeline: ERA5 provides daily/hourly snapshots at exactly T, enabling strict historical replay

### Variables selected from weather sources

From ECMWF IFS Open Data / GFS (forecast horizon ≤ 72 h initially):
- `u10`, `v10` — 10 m wind components (derive speed and direction)
- `msl` — mean sea level pressure
- `swh` — significant wave height (from coupled wave model)
- `t2m` — 2 m air temperature

From ERA5 (same variables, for training data):
- Same as above, plus sea ice concentration (`siconc`) as cross-validation against OSI SAF

---

## Part 4 — Ocean Data

### Candidates evaluated

1. **CMEMS PHY 001_024** (Global Ocean Physics Analysis and Forecast)
2. **CMEMS GLORYS12** (Global Ocean Physics Reanalysis)
3. **OSCAR v2.0** (Ocean Surface Current Analysis Real-time)
4. **HYCOM** (HYbrid Coordinate Ocean Model, US Navy)

### Comparison table

| Criterion | CMEMS PHY Forecast | GLORYS12 | OSCAR v2.0 | HYCOM |
|-----------|-------------------|----------|------------|-------|
| Type | Analysis + 10-day forecast | Ocean reanalysis | L4 obs-derived analysis | Analysis + ~7-day forecast |
| Resolution | **1/12° ≈ 8 km** | 1/12° ≈ 8 km | 0.25° | 1/12° at equator |
| Antarctic coverage | **80°S–90°N** | 80°S–90°N | Global (degrades > 60°S) | **Hard limit: 78°S** |
| History | From Nov 2020 | **1993–present** | 1993–present (NRT: 2-day lag) | Last ~10–14 days on THREDDS |
| Forecast horizon | **10 days** | None (reanalysis) | None (2-day latency NRT) | 5–7 days |
| Key variables | uo, vo, thetao, sithick, siconc, SSH | Same as forecast product | U, V surface only | Temp, Sal, uo, vo, SSH |
| Auth | Free CMEMS account | Same | NASA Earthdata account | None (OPeNDAP) |
| Python library | `copernicusmarine` | Same library | `earthaccess` | `xarray` OPeNDAP |
| Automation | Low | Low | Low-medium | Medium (URL changes with model update) |
| Licensing | Copernicus (free) | Copernicus (free) | NASA (free) | US public domain |

### Decision

**Primary ocean forecast (operational):** CMEMS PHY 001_024  
Product ID: `GLOBAL_ANALYSISFORECAST_PHY_001_024`  
Dataset: `cmems_mod_glo_phy-all_anfc_0.083deg_P1D-m`  
- 8 km resolution, full Southern Ocean coverage to 80°S  
- 10-day forecast updated daily — longest ocean forecast horizon of any free source  
- Provides `uo`, `vo` (currents), `thetao` (SST), `siconc`, `sithick` all in one product  
- The `copernicusmarine.subset()` bounding-box API returns xarray directly; no volume quotas

**Historical training / benchmark:** CMEMS GLORYS12  
Product ID: `GLOBAL_MULTIYEAR_PHY_001_030`  
Dataset: `cmems_mod_glo_phy_my_0.083deg_P1D-m`  
- Same variable set and API as the forecast product; one account, one library, consistent variables
- Daily mean from 1993 — adequate for historical replay with strict time-gating  
- Monthly mean also available for faster exploratory training runs

**Not selected:** OSCAR v2.0 — useful as independent cross-check, but (a) no forecast capability, (b) minimum 2-day NRT latency, (c) accuracy degrades south of ~60°S due to sparse altimetry satellite coverage.

**Rejected:** HYCOM — hard southern limit at 78°S disqualifies it for full Antarctic routing coverage.

### Variables selected from ocean sources

From CMEMS PHY 001_024 / GLORYS12:
- `uo`, `vo` — eastward and northward ocean current velocity at surface level
- `thetao` [surface] — sea surface temperature
- `siconc` — sea ice concentration (cross-check with OSI SAF primary)
- `sithick` — sea ice thickness (only source providing this; not available from OSI SAF)
- `mlotst` — mixed layer depth (used in iceberg melt/drift physics features)

---

## Part 5 — Final Recommended Provider Stack

### Minimum viable for SIH MVP

```
Sea ice    →  OSI SAF (primary NRT)  +  NSIDC G02135 (fallback/extent)
Icebergs   →  USNIC CSV (operational positions)  +  BYU/NIC SCP (ML training)
Weather    →  ECMWF Open Data IFS (primary)  +  GFS via NOMADS (fallback)
Ocean      →  CMEMS PHY 001_024 (operational)  +  GLORYS12 (historical)
```

### Account / authentication requirements

| Service | Account | Where to register | What it unlocks |
|---------|---------|-------------------|-----------------|
| CMEMS | Free, ~2 min | `https://marine.copernicus.eu/` | GLORYS12 + PHY forecast + NRT sea ice |
| CDS (ERA5) | Free | `https://cds.climate.copernicus.eu/` | ERA5 reanalysis |
| ECMWF Open Data | Free | `https://www.ecmwf.int/` | IFS full-res forecast |
| NASA Earthdata | Free (optional) | `https://urs.earthdata.nasa.gov/` | OSCAR validation only |
| **None** | — | — | GFS, USNIC, NSIDC G02135, BYU/NIC SCP |

### Python dependencies (cloud component additions for Phase 2+)

```
copernicusmarine>=2.0     # CMEMS (ocean + sea ice NRT)
cdsapi>=0.7               # ERA5
ecmwf-opendata>=0.3       # ECMWF IFS open data
herbie-data>=2024.9       # GFS via NOMADS
xarray>=2024.3            # Common data layer
cfgrib>=0.9               # GRIB2 reading (ECMWF/GFS)
scipy>=1.13               # Regridding
pandas>=2.2               # Iceberg CSV processing
```

### Credentials file layout (runtime, not in repo)

```
~/.cdsapirc             # ERA5: url + key
~/.copernicusmarine/    # Created by: copernicusmarine login
~/.ecmwf/              # Created by: ecmwf-opendata login  
env: EARTHDATA_USERNAME / EARTHDATA_PASSWORD  # OSCAR (optional)
```

---

## Part 6 — ML Model Variable Specification

### Model A: Sea Ice Forecasting

**Objective:** Predict sea ice concentration at 6h, 12h, 24h, 48h, 72h horizons on the ForecastPackage 20×20 grid.

**Input features (per grid cell, per time step):**
- `siconc_t0` — current sea ice concentration (OSI SAF)
- `siconc_t-1d`, `siconc_t-3d`, `siconc_t-7d` — recent history
- `sithick_t0` — current sea ice thickness (CMEMS)
- `sst_t0` — sea surface temperature (CMEMS)
- `u10_t0`, `v10_t0` — 10 m wind (ECMWF IFS)
- `uo_t0`, `vo_t0` — ocean surface current (CMEMS)
- `month`, `lat`, `lon` — spatial/temporal position encoding

**Output:**
- `siconc_hat_tH` — predicted concentration at horizon H
- `siconc_unc_tH` — predicted uncertainty (aleatoric)

**Training data:** OSI SAF CDR/ICDR concentration (1978–present) + GLORYS12 ocean + ERA5 weather

### Model B: Iceberg Trajectory Prediction

**Objective:** Predict iceberg position at 6h, 12h, 24h, 48h, 72h given current position and environment.

**Input features (per iceberg, per time step):**
- `lat_t0`, `lon_t0` — current position (USNIC / BYU/NIC)
- `length_nm`, `width_nm` — size (USNIC)
- `uo_t0`, `vo_t0` — ocean surface current at iceberg position (CMEMS)
- `u10_t0`, `v10_t0` — surface wind at iceberg position (ECMWF IFS)
- `siconc_t0` — sea ice concentration at iceberg position (OSI SAF)
- `sithick_t0` — sea ice thickness at position (CMEMS)
- `mld_t0` — mixed layer depth (CMEMS — affects ocean drag)
- `doy` — day of year (seasonal signal)

**Output:**
- `lat_hat_tH`, `lon_hat_tH` — predicted position at horizon H
- `uncertainty_radius_km_tH` — position uncertainty (grows with horizon)

**Training data:** BYU/NIC SCP daily positions (1978–2023) + community S3 mirror (2014–present) + matched CMEMS/ERA5 forcing fields

**Reference approach:** IDRIFTNET (arXiv:2507.00036) — physics-informed residual learning where Lagrangian ocean advection provides the physical prior and the network learns the residual drift (wind drag, wave-induced Stokes drift, keel-depth correction).

---

## Part 7 — Data Pipeline Architecture

### High-level flow

```
┌─────────────────────────────────────────────────────────────────┐
│  CLOUD COMPONENT — cloud/data/                                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  sea_ice/    │  │  icebergs/   │  │  environment/         │ │
│  │  connector   │  │  connector   │  │  connector            │ │
│  │              │  │              │  │                       │ │
│  │  OSI SAF     │  │  USNIC CSV   │  │  ECMWF Open Data      │ │
│  │  (primary)   │  │  (primary)   │  │  (weather primary)    │ │
│  │              │  │              │  │                       │ │
│  │  NSIDC G02135│  │  BYU/NIC SCP │  │  GFS via herbie       │ │
│  │  (fallback)  │  │  (train only)│  │  (weather fallback)   │ │
│  └──────┬───────┘  └──────┬───────┘  │                       │ │
│         │                 │          │  CMEMS PHY 001_024     │ │
│         │                 │          │  (ocean primary)       │ │
│         │                 │          └───────────┬───────────┘ │
│         │                 │                      │             │
│         └─────────────────┴──────────────────────┘             │
│                           │                                     │
│              ┌────────────▼────────────┐                        │
│              │   assembler.py          │                        │
│              │   build_forecast_       │                        │
│              │   package(request)      │                        │
│              │   → ForecastPackage     │                        │
│              └────────────┬────────────┘                        │
│                           │                                     │
│              POST /forecast → ForecastPackage (JSON)            │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                    (vessel pulls)
┌───────────────────────────▼─────────────────────────────────────┐
│  VESSEL — local cache + navigation intelligence                  │
└─────────────────────────────────────────────────────────────────┘
```

### Module layout for cloud/data/ (Phase 2 implementation)

```
cloud/data/
├── sea_ice/
│   ├── __init__.py
│   ├── osisaf.py        # OSI SAF CDR/NRT fetcher (primary)
│   └── nsidc.py         # NSIDC G02135 fetcher (fallback / extent)
├── icebergs/
│   ├── __init__.py
│   ├── usnic.py         # USNIC weekly CSV fetcher (operational)
│   └── training.py      # BYU/NIC SCP + S3 mirror (training data only)
├── environment/
│   ├── __init__.py
│   ├── weather.py       # ECMWF Open Data + GFS fallback
│   └── ocean.py         # CMEMS PHY 001_024 + GLORYS12
├── regrid.py            # Common regridding to ForecastPackage grid
└── assembler.py         # Coordinates all connectors → ForecastPackage
```

### Regridding note

All sources use different grids (polar-stereo, lat-lon, Gaussian). The assembler must reproject all inputs to the ForecastPackage's simple lat-lon grid before assembly. `scipy.interpolate.RegularGridInterpolator` or `xarray`'s `.interp()` are adequate for MVP at 1° or 0.5° resolution. No binary formats introduced — output stays `list[list[float]]` per the Phase 1 schema.

### Benchmark pipeline extensions

For strict time-gating (per CLAUDE.md design rule):
- ERA5: query with `date=T` — hourly reanalysis at exactly timestamp T
- GLORYS12: query with `time=T` — daily mean; use floor(T) as the valid timestamp
- OSI SAF CDR: file naming is date-based; select file where `valid_time ≤ T`
- USNIC archive: select the snapshot with the latest `Last Update ≤ T`
- BYU/NIC: select iceberg positions where `observation_date ≤ T`

No data with a timestamp > T ever enters the replay system at time T.

---

## Part 8 — Proof-of-Access Summary

Direct network tests were blocked by the managed build environment's outbound proxy policy. Access patterns are documented from:
1. Official provider documentation (fetched by research agents via WebFetch)
2. Published Python library documentation and example code
3. Third-party tooling repositories (herbie, copernicusmarine, earthaccess)

The following no-authentication endpoints should be verified locally before Phase 2 implementation begins:

```bash
# NSIDC G02135 — zero auth daily extent CSV
curl -L "https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v4.0.csv" | head -5

# USNIC current iceberg positions
curl -L "https://usicecenter.gov/File/DownloadCurrent?pId=134" | head -5

# Community S3 mirror (aggregated USNIC)
curl -L "https://usi-icebergs.s3.eu-central-1.amazonaws.com/icebergs_locations_usi.csv" | head -5

# NOMADS GFS catalog (no auth)
curl "https://nomads.ncep.noaa.gov/dods/" | grep gfs_0p25

# OSI SAF THREDDS catalog (no auth for CDR)
curl "https://thredds.met.no/thredds/catalog/osisaf/met.no/reprocessed/ice/conc/v3p1/catalog.html" | grep -c "\.nc"
```

---

## Part 9 — Provider Decisions Not Made

The following architectural decisions are explicitly deferred:

1. **Sea ice ML architecture** — not locked (ConvLSTM, U-Net, SwinTransformer, or simple statistical baseline); evaluate after training data pipeline is established
2. **Iceberg ML architecture** — IDRIFTNET referenced as a guide but not mandated; baseline Lagrangian advection must be benchmarked first
3. **Exact ECMWF vs GFS selection** — both in the stack; runtime selection based on availability
4. **OSI SAF NRT access path** — either MET Norway THREDDS (no auth) or CMEMS NRT layer (one account); decision on access stability after first live test
5. **Regridding resolution** — ForecastPackage 20×20 grid at ±10° ≈ 1° resolution; may be refined once ML model input requirements are clearer
