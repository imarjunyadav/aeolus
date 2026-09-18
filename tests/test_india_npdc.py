from datetime import datetime, timezone

import pytest

from cloud.data.india.npdc import NpdcParseError, parse_npdc_csv, parse_npdc_json


class TestNpdcAdapter:
    def test_csv_normalization(self):
        text = "timestamp,station,temp,rh,pressure,wind_speed,wind_direction\n2026-09-17T00:00:00Z,Maitri,-18.5,72,982.4,7.2,245\n"
        snapshots = parse_npdc_csv(text, datetime(2026, 9, 17, 1, tzinfo=timezone.utc))
        snap = snapshots[0]
        assert snap.air_temperature_c == -18.5
        assert snap.pressure_hpa == 982.4
        assert snap.wind_speed_ms == 7.2
        assert snap.metadata["station"] == "Maitri"
        assert snap.metadata["relative_humidity_pct"] == 72
        assert snap.source_data_age_seconds == 3600

    def test_json_normalization(self):
        payload = {"observations": [{"datetime": "2026-09-17T00:00:00Z", "station_name": "Bharati", "temperature_c": -12}]}
        snap = parse_npdc_json(payload)[0]
        assert snap.air_temperature_c == -12
        assert snap.metadata["station"] == "Bharati"

    def test_invalid_rows_are_rejected(self):
        with pytest.raises(NpdcParseError):
            parse_npdc_csv("timestamp,station,temp\nnot-a-date,Maitri,foo\n")
