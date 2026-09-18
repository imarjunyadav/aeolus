# Phase 4 — ML foundation

Aeolus now has a provider-agnostic dataset builder and two transparent forecasting baselines.

## Sea ice

`cloud/models/sea_ice_dataset.py` turns an xarray DataArray/Dataset with a `time` dimension into chronological supervised windows:

- inputs: recent concentration frames `[samples, input_steps, lat, lon]`
- target: future concentration `[samples, lat, lon]`
- missing cells are filtered/filled conservatively
- percentage inputs are normalized to fractions
- source cadence is preserved instead of assuming a fixed 6-hour interval

The first trainable model is an intentionally small per-grid-cell linear-trend forecaster in `sea_ice_forecaster.py`. It is **not yet the production architecture**. It exists so we can establish whether a learned model beats persistence before increasing model complexity. A deep-learning implementation can be evaluated later after dataset inspection.

No additional ML framework is required for this first baseline.

## Icebergs

`cloud/models/iceberg_baseline.py` implements a constant-velocity trajectory baseline from the latest two observations. The eventual learned trajectory model must beat this baseline on held-out tracks.

## Scientific evaluation rule

Do not train/test by randomly shuffling adjacent frames. Split by time and, where possible, by spatial region/track so that nearby observations do not leak between train and test.

The next dataset work should produce a benchmark table containing persistence MAE and learned-model MAE at each forecast horizon (6/12/24/48/72h where the source cadence supports them).


## Benchmark command

After obtaining historical Antarctic sea-ice NetCDF files:

```bash
python tools/benchmark_ml.py /path/to/file1.nc /path/to/file2.nc \
  --variable ice_conc --input-steps 4 --lead-steps 1 --output artifacts/sea_ice_benchmark.json
```

The split is chronological: the newest 20% of supervised windows are held
out. The report compares persistence MAE against the linear-trend baseline.
Run separate evaluations for each supported forecast horizon rather than
combining horizons into one score.

The benchmark is intentionally dataset-driven. Do not report a model as
operationally useful until it has been evaluated on held-out historical data.
