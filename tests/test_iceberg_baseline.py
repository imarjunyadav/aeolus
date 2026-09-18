from datetime import datetime, timezone, timedelta
from cloud.models.iceberg_baseline import TrackPoint, constant_velocity_forecast


def test_constant_velocity_trajectory():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [TrackPoint(t, -70.0, 10.0), TrackPoint(t + timedelta(hours=1), -69.0, 11.0)]
    result = constant_velocity_forecast(history, [6, 12])
    assert result[0].latitude == -63.0
    assert result[0].longitude == 17.0
    assert result[1].latitude == -57.0
    assert result[1].longitude == 23.0
