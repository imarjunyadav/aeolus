"""Utilities for turning gridded sea-ice observations into ML-ready sequences.

The dataset builder is deliberately provider-agnostic: it accepts an xarray
DataArray or Dataset with a time dimension and produces chronological
input/target windows. No model choice is baked into this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class SeaIceSequence:
    """One supervised forecasting example."""

    inputs: np.ndarray  # [input_steps, lat, lon]
    target: np.ndarray  # [lat, lon]
    target_time: np.datetime64


def _as_data_array(source: xr.DataArray | xr.Dataset, variable: str | None) -> xr.DataArray:
    if isinstance(source, xr.DataArray):
        da = source
    else:
        if variable is None:
            candidates = [name for name, value in source.data_vars.items() if "time" in value.dims and value.ndim >= 3]
            if not candidates:
                raise ValueError("Dataset contains no suitable time-varying gridded variable")
            variable = candidates[0]
        if variable not in source.data_vars:
            raise KeyError(f"Variable {variable!r} not found in dataset")
        da = source[variable]
    if "time" not in da.dims:
        raise ValueError("Sea-ice data must contain a 'time' dimension")
    spatial = [d for d in da.dims if d != "time"]
    if len(spatial) != 2:
        raise ValueError("Sea-ice data must have exactly two spatial dimensions")
    return da.transpose("time", spatial[0], spatial[1])


def clean_concentration(da: xr.DataArray) -> xr.DataArray:
    """Normalize concentration to [0, 1] and preserve missing cells as NaN."""
    values = da.astype("float32")
    # Common provider convention: percentage values 0..100.
    finite = values.values[np.isfinite(values.values)]
    if finite.size and float(np.nanpercentile(finite, 99)) > 1.5:
        values = values / 100.0
    values = values.where((values >= 0.0) & (values <= 1.0))
    return values


def build_sequences(
    source: xr.DataArray | xr.Dataset,
    *,
    input_steps: int = 4,
    lead_steps: int = 1,
    variable: str | None = None,
    drop_missing_fraction: float = 0.20,
) -> list[SeaIceSequence]:
    """Create chronological sliding windows from gridded sea-ice data.

    ``lead_steps=1`` means the target is the next available observation. This
    avoids assuming that every provider has a fixed 6-hour cadence; cadence is
    measured from the source timestamps instead.
    """
    if input_steps < 1 or lead_steps < 1:
        raise ValueError("input_steps and lead_steps must be positive")
    da = clean_concentration(_as_data_array(source, variable)).sortby("time")
    arr = np.asarray(da.values, dtype=np.float32)
    times = np.asarray(da.time.values)
    sequences: list[SeaIceSequence] = []
    end = input_steps + lead_steps - 1
    for target_idx in range(input_steps, len(times) - lead_steps + 1):
        x = arr[target_idx - input_steps : target_idx]
        y = arr[target_idx + lead_steps - 1]
        missing = float(np.mean(~np.isfinite(np.concatenate([x.reshape(-1), y.reshape(-1)]))))
        if missing > drop_missing_fraction:
            continue
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        sequences.append(SeaIceSequence(x, y, times[target_idx + lead_steps - 1]))
    return sequences


def load_netcdf_sequences(
    paths: str | Path | Iterable[str | Path],
    *,
    variable: str | None = None,
    input_steps: int = 4,
    lead_steps: int = 1,
) -> list[SeaIceSequence]:
    """Load one or more NetCDF files and build sequences in time order."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = [Path(p) for p in paths]
    if not files:
        raise ValueError("No NetCDF files supplied")
    datasets = [xr.open_dataset(path) for path in files]
    try:
        combined = xr.concat(datasets, dim="time").sortby("time")
        return build_sequences(combined, variable=variable, input_steps=input_steps, lead_steps=lead_steps)
    finally:
        for ds in datasets:
            ds.close()
