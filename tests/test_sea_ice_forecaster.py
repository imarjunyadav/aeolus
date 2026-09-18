import numpy as np
from cloud.models.sea_ice_forecaster import LinearTrendSeaIceForecaster


def test_linear_trend_forecaster_learns_positive_trend():
    # One sample, four historical frames, one grid cell.
    base = np.array([[[[0.1]], [[0.2]], [[0.3]], [[0.4]]]], dtype=float)
    model = LinearTrendSeaIceForecaster(input_steps=4).fit(base)
    pred = model.predict(base, lead_steps=2)
    assert pred.shape == (1, 1, 1)
    assert np.allclose(pred[:, 0, 0], 0.6)


def test_linear_trend_forecaster_clips_to_physical_range():
    x = np.array([[[[0.8, 0.9]], [[0.9, 1.0]]]], dtype=float)
    model = LinearTrendSeaIceForecaster(2).fit(x)
    pred = model.predict(x, lead_steps=10)
    assert np.all((pred >= 0.0) & (pred <= 1.0))
