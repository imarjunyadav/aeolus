# Aeolus current state

## Verified locally from the current source tree

The project contains working Phase-3 provider connectors, a real assembler,
NPDC ingestion, a deterministic navigation MVP, and a Phase-4 baseline/data
pipeline.

Run:

```bash
python -m pytest -q
```

In the build environment used for this review, 324 tests passed. Two live
provider tests fail only because `copernicusmarine` and `pydap` are not
installed in that environment; the project requirements list both.

## Next evidence-producing step

Use `tools/benchmark_ml.py` with historical Antarctic sea-ice NetCDF data.
Produce persistence-vs-trend MAE by forecast horizon before selecting a
deep-learning architecture.

See `docs/placeholders.md` for every remaining placeholder or development-only
component that must be replaced before operational deployment.
