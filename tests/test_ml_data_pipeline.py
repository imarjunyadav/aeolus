import numpy as np
import xarray as xr

from cloud.models.baseline import mae, persistence_forecast
from cloud.models.sea_ice_dataset import build_sequences, clean_concentration


def test_sea_ice_sequence_builder_and_normalization():
    data = np.stack([np.full((3, 4), i * 10.0) for i in range(8)])
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": np.arange(8)})
    cleaned = clean_concentration(da)
    assert np.isclose(float(cleaned.max()), 0.7)
    seq = build_sequences(da, input_steps=3, lead_steps=2)
    assert len(seq) == 4
    assert seq[0].inputs.shape == (3, 3, 4)
    assert seq[0].target.shape == (3, 4)


def test_persistence_is_benchmarkable():
    actual = np.full((2, 2), 0.7)
    pred = persistence_forecast(np.full((2, 2), 0.5), [6])[0]
    assert np.isclose(mae(actual, pred), 0.2)
