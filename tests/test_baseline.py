import numpy as np
from cloud.models.baseline import mae, persistence_forecast


def test_persistence_baseline_and_mae():
    grid = np.array([[0.1, 0.2], [0.3, 0.4]])
    preds = persistence_forecast(grid, [6, 12])
    assert len(preds) == 2
    assert np.array_equal(preds[0], grid)
    assert mae(grid, preds[0]) == 0.0
