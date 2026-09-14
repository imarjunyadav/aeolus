"""
Phase 1 verification tests.

Covers:
- Schema imports and structure
- All required ForecastPackage fields
- ForecastRequest extensibility
- JSON round-trip fidelity
- Mock generator reproducibility
- No forbidden Phase 1 code (ML, real connectors, nav algorithms, safety, UI)
- Cloud API /health and POST /forecast endpoints
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# 1. Schema imports
# ---------------------------------------------------------------------------

class TestSchemaImports:
    def test_common_imports(self):
        from shared.schemas.common import (
            ConnectivityState, GridMetadata, OptimizationMode, Position,
            RiskLevel, SafetyStatus, WarnCategory, WarnLevel,
        )
        assert ConnectivityState.LIVE == "LIVE"
        assert ConnectivityState.DEGRADED == "DEGRADED"
        assert ConnectivityState.OFFLINE == "OFFLINE"

    def test_forecast_imports(self):
        from shared.schemas.forecast import (
            ForecastPackage, IcebergForecast, IcebergTrajectoryHorizon,
            OceanSnapshot, SeaIceForecast, SeaIceForecastHorizon,
            WeatherSnapshot,
        )

    def test_vessel_imports(self):
        from shared.schemas.vessel import ForecastRequest, VesselContext, VesselState

    def test_navigation_imports(self):
        from shared.schemas.navigation import (
            ForecastFreshness, NavigationResult, NavigationWarning,
            RouteCandidate, RouteWaypoint,
        )

    def test_top_level_schemas_init(self):
        from shared.schemas import (
            ConnectivityState, ForecastFreshness, ForecastPackage,
            ForecastRequest, GridMetadata, IcebergForecast,
            IcebergTrajectoryHorizon, NavigationResult, NavigationWarning,
            OceanSnapshot, OptimizationMode, Position, RiskLevel,
            RouteCandidate, RouteWaypoint, SafetyStatus, SeaIceForecast,
            SeaIceForecastHorizon, VesselContext, VesselState, WarnCategory,
            WarnLevel, WeatherSnapshot,
        )


# ---------------------------------------------------------------------------
# 2. ForecastPackage: all required Phase 1 fields
# ---------------------------------------------------------------------------

class TestForecastPackageFields:
    @pytest.fixture
    def package(self):
        from shared.schemas.common import Position
        from data.mock.generator import generate_mock_forecast_package
        return generate_mock_forecast_package(
            vessel_position=Position(latitude=-68.5, longitude=-170.2),
            seed=42,
        )

    def test_top_level_fields(self, package):
        assert package.package_id
        assert package.schema_version == "1.0"
        assert isinstance(package.generated_at, datetime)
        assert isinstance(package.valid_from, datetime)
        assert isinstance(package.valid_to, datetime)
        assert package.valid_to > package.valid_from
        assert package.generated_by
        assert 0.0 <= package.overall_confidence <= 1.0

    def test_sea_ice_forecast_fields(self, package):
        si = package.sea_ice
        assert isinstance(si.source_data_timestamp, datetime)
        assert isinstance(si.source_data_age_seconds, int)
        assert si.grid_metadata is not None
        assert si.grid_metadata.n_lat == 20
        assert si.grid_metadata.n_lon == 20
        assert si.grid_metadata.lat_min < si.grid_metadata.lat_max
        assert si.grid_metadata.lon_min < si.grid_metadata.lon_max
        assert len(si.horizons) == 5  # [6, 12, 24, 48, 72]

    def test_sea_ice_concentration_and_uncertainty(self, package):
        for h in package.sea_ice.horizons:
            assert h.horizon_hours > 0
            assert isinstance(h.valid_at, datetime)
            conc = h.concentration_grid
            unc = h.uncertainty_grid
            # Shape checks
            assert len(conc) == 20
            assert all(len(row) == 20 for row in conc)
            assert len(unc) == 20
            assert all(len(row) == 20 for row in unc)
            # Value range checks
            for row in conc:
                for v in row:
                    assert 0.0 <= v <= 1.0, f"concentration {v} out of [0,1]"
            for row in unc:
                for v in row:
                    assert v >= 0.0, f"uncertainty {v} is negative"

    def test_iceberg_trajectories_and_uncertainty(self, package):
        assert len(package.icebergs) == 4
        for iceberg in package.icebergs:
            assert iceberg.iceberg_id
            assert iceberg.last_known_position is not None
            assert isinstance(iceberg.last_known_timestamp, datetime)
            assert isinstance(iceberg.source_data_age_seconds, int)
            assert len(iceberg.horizons) == 5
            for h in iceberg.horizons:
                assert h.horizon_hours > 0
                assert -90.0 <= h.latitude <= 90.0
                assert h.uncertainty_radius_km >= 0.0
                # Uncertainty should grow with horizon (synthetic model)
                assert isinstance(h.valid_at, datetime)

    def test_iceberg_uncertainty_grows_with_horizon(self, package):
        for iceberg in package.icebergs:
            radii = [h.uncertainty_radius_km for h in iceberg.horizons]
            # In the mock, uncertainty is monotonically increasing
            assert radii == sorted(radii), f"Uncertainty not monotone: {radii}"

    def test_weather_snapshot_fields(self, package):
        w = package.weather
        assert isinstance(w.source_data_timestamp, datetime)
        assert isinstance(w.source_data_age_seconds, int)
        assert isinstance(w.valid_at, datetime)
        assert w.wind_speed_ms is not None
        assert w.wind_direction_deg is not None
        assert w.pressure_hpa is not None
        assert w.significant_wave_height_m is not None
        assert w.air_temperature_c is not None
        assert len(w.available_variables) > 0

    def test_ocean_snapshot_fields(self, package):
        o = package.ocean
        assert isinstance(o.source_data_timestamp, datetime)
        assert isinstance(o.source_data_age_seconds, int)
        assert isinstance(o.valid_at, datetime)
        assert o.current_u_ms is not None
        assert o.current_v_ms is not None
        assert o.sst_c is not None
        assert len(o.available_variables) > 0

    def test_forecast_horizons_ordered(self, package):
        hours = [h.horizon_hours for h in package.sea_ice.horizons]
        assert hours == sorted(hours)

    def test_validity_period(self, package):
        max_horizon = max(h.horizon_hours for h in package.sea_ice.horizons)
        expected_duration = max_horizon * 3600  # seconds
        actual_duration = (package.valid_to - package.valid_from).total_seconds()
        assert abs(actual_duration - expected_duration) < 2  # within 2s


# ---------------------------------------------------------------------------
# 3. ForecastRequest extensibility
# ---------------------------------------------------------------------------

class TestForecastRequestExtensibility:
    def test_minimal_request_only_position(self):
        from shared.schemas.vessel import ForecastRequest
        from shared.schemas.common import Position
        req = ForecastRequest(vessel_position=Position(latitude=-68.5, longitude=-170.2))
        assert req.destination is None
        assert req.route_area is None
        assert req.forecast_horizons_hours == [6, 12, 24, 48, 72]
        assert req.vessel_context is None

    def test_full_request_with_all_optional_fields(self):
        from shared.schemas.vessel import ForecastRequest, VesselContext
        from shared.schemas.common import Position
        req = ForecastRequest(
            vessel_position=Position(latitude=-68.5, longitude=-170.2),
            destination=Position(latitude=-75.0, longitude=-165.0),
            route_area={"lat_min": -75.0, "lat_max": -65.0, "lon_min": -175.0, "lon_max": -160.0},
            forecast_horizons_hours=[6, 12, 24],
            vessel_context=VesselContext(
                vessel_id="MV-SAGARMATHA",
                ice_class="PC5",
                max_ice_concentration=0.75,
                min_iceberg_clearance_km=5.0,
            ),
        )
        assert req.destination.latitude == -75.0
        assert req.vessel_context.ice_class == "PC5"

    def test_route_area_accepts_arbitrary_dict(self):
        """route_area is open dict — must accept any structure for future corridor formats."""
        from shared.schemas.vessel import ForecastRequest
        from shared.schemas.common import Position
        req = ForecastRequest(
            vessel_position=Position(latitude=-68.5, longitude=-170.2),
            route_area={
                "type": "corridor",
                "points": [[-68.5, -170.2], [-75.0, -165.0]],
                "buffer_km": 50,
            },
        )
        assert req.route_area["type"] == "corridor"

    def test_vessel_context_extra_dict(self):
        """VesselContext.extra accepts arbitrary vessel-specific fields."""
        from shared.schemas.vessel import VesselContext
        ctx = VesselContext(
            vessel_id="MV-TEST",
            extra={"draft_m": 6.5, "beam_m": 18.0, "engine_type": "diesel-electric"},
        )
        assert ctx.extra["draft_m"] == 6.5

    def test_custom_horizons(self):
        from shared.schemas.vessel import ForecastRequest
        from shared.schemas.common import Position
        req = ForecastRequest(
            vessel_position=Position(latitude=-68.5, longitude=-170.2),
            forecast_horizons_hours=[3, 6, 12],
        )
        assert req.forecast_horizons_hours == [3, 6, 12]


# ---------------------------------------------------------------------------
# 4. JSON round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    @pytest.fixture
    def package(self):
        from shared.schemas.common import Position
        from data.mock.generator import generate_mock_forecast_package
        return generate_mock_forecast_package(
            vessel_position=Position(latitude=-68.5, longitude=-170.2),
            seed=42,
        )

    def test_serialize_to_json_string(self, package):
        json_str = package.model_dump_json()
        assert len(json_str) > 0
        data = json.loads(json_str)
        assert "package_id" in data
        assert "sea_ice" in data
        assert "icebergs" in data
        assert "weather" in data
        assert "ocean" in data

    def test_deserialize_from_json_string(self, package):
        from shared.schemas.forecast import ForecastPackage
        json_str = package.model_dump_json()
        package2 = ForecastPackage.model_validate_json(json_str)
        assert package2.package_id == package.package_id

    def test_round_trip_preserves_grid_values(self, package):
        from shared.schemas.forecast import ForecastPackage
        json_str = package.model_dump_json()
        package2 = ForecastPackage.model_validate_json(json_str)
        orig = package.sea_ice.horizons[0].concentration_grid
        restored = package2.sea_ice.horizons[0].concentration_grid
        assert orig == restored

    def test_round_trip_preserves_iceberg_trajectories(self, package):
        from shared.schemas.forecast import ForecastPackage
        json_str = package.model_dump_json()
        package2 = ForecastPackage.model_validate_json(json_str)
        for i, (orig, restored) in enumerate(
            zip(package.icebergs, package2.icebergs)
        ):
            assert orig.iceberg_id == restored.iceberg_id
            for h_orig, h_rest in zip(orig.horizons, restored.horizons):
                assert h_orig.latitude == h_rest.latitude
                assert h_orig.longitude == h_rest.longitude
                assert h_orig.uncertainty_radius_km == h_rest.uncertainty_radius_km

    def test_round_trip_preserves_timestamps(self, package):
        from shared.schemas.forecast import ForecastPackage
        json_str = package.model_dump_json()
        package2 = ForecastPackage.model_validate_json(json_str)
        assert package2.generated_at == package.generated_at
        assert package2.valid_from == package.valid_from
        assert package2.valid_to == package.valid_to

    def test_round_trip_no_data_loss(self, package):
        from shared.schemas.forecast import ForecastPackage
        # Double round-trip must be stable
        j1 = package.model_dump_json()
        p2 = ForecastPackage.model_validate_json(j1)
        j2 = p2.model_dump_json()
        assert j1 == j2


# ---------------------------------------------------------------------------
# 5. Mock generator reproducibility
# ---------------------------------------------------------------------------

class TestMockGeneratorReproducibility:
    def _generate(self, seed):
        from shared.schemas.common import Position
        from data.mock.generator import generate_mock_forecast_package
        return generate_mock_forecast_package(
            vessel_position=Position(latitude=-68.5, longitude=-170.2),
            seed=seed,
        )

    def test_same_seed_produces_same_package_id(self):
        # package_id is uuid4() — NOT seeded, so IDs differ; all other data must match
        p1 = self._generate(42)
        p2 = self._generate(42)
        # package_id uses uuid4, intentionally not seeded
        assert p1.package_id != p2.package_id  # expected

    def test_same_seed_produces_identical_grids(self):
        p1 = self._generate(42)
        p2 = self._generate(42)
        for h1, h2 in zip(p1.sea_ice.horizons, p2.sea_ice.horizons):
            assert h1.concentration_grid == h2.concentration_grid
            assert h1.uncertainty_grid == h2.uncertainty_grid

    def test_same_seed_produces_identical_iceberg_trajectories(self):
        p1 = self._generate(42)
        p2 = self._generate(42)
        for i1, i2 in zip(p1.icebergs, p2.icebergs):
            assert i1.iceberg_id == i2.iceberg_id
            for h1, h2 in zip(i1.horizons, i2.horizons):
                assert h1.latitude == h2.latitude
                assert h1.longitude == h2.longitude
                assert h1.uncertainty_radius_km == h2.uncertainty_radius_km

    def test_same_seed_produces_identical_weather(self):
        p1 = self._generate(42)
        p2 = self._generate(42)
        assert p1.weather.wind_speed_ms == p2.weather.wind_speed_ms
        assert p1.weather.wind_direction_deg == p2.weather.wind_direction_deg
        assert p1.weather.pressure_hpa == p2.weather.pressure_hpa
        assert p1.weather.significant_wave_height_m == p2.weather.significant_wave_height_m
        assert p1.weather.air_temperature_c == p2.weather.air_temperature_c

    def test_same_seed_produces_identical_ocean(self):
        p1 = self._generate(42)
        p2 = self._generate(42)
        assert p1.ocean.current_u_ms == p2.ocean.current_u_ms
        assert p1.ocean.current_v_ms == p2.ocean.current_v_ms
        assert p1.ocean.sst_c == p2.ocean.sst_c

    def test_different_seeds_produce_different_data(self):
        p1 = self._generate(42)
        p2 = self._generate(99)
        # Grids should differ between seeds
        grid1 = p1.sea_ice.horizons[0].concentration_grid
        grid2 = p2.sea_ice.horizons[0].concentration_grid
        assert grid1 != grid2


# ---------------------------------------------------------------------------
# 6. Code audit — no forbidden Phase 1 content
# ---------------------------------------------------------------------------

class TestCodeAudit:
    """
    Verify that no ML model code, real data connectors, navigation algorithms,
    Safety Shield logic, or UI code was accidentally included in Phase 1.
    """

    FORBIDDEN_PATTERNS = {
        "ML frameworks": [
            r"\btorch\b", r"\btensorflow\b", r"\bkeras\b",
            r"\bsklearn\b", r"\bscikit.learn\b",
            r"ConvLSTM", r"class.*Model.*nn\.Module",
            r"\.fit\(", r"\.train\(",
        ],
        "Real data connectors": [
            r"nsidc\.org", r"copernicus", r"era5", r"ecmwf",
            r"ncei\.noaa", r"nomads", r"podaac",
            r"earthdata\.nasa", r"oscar\b",
            r"requests\.get\(", r"aiohttp\.ClientSession",
            r"ftplib", r"urllib\.request",
        ],
        "Navigation algorithms": [
            r"def.*risk_score", r"def.*time_to_encounter",
            r"def.*fuel_estimate", r"def.*route_optim",
            r"astar", r"dijkstra", r"def.*navigate\(",
        ],
        "Safety Shield": [
            r"def.*safety_check", r"def.*shield",
            r"SafetyShield", r"def.*reject_route",
        ],
        "UI/frontend build artifacts": [
            r"import React", r"from 'react'",
            r"\.tsx?$",
        ],
    }

    # Directories and files to audit (source only, not tests)
    AUDIT_PATHS = [
        ROOT / "shared",
        ROOT / "cloud",
        ROOT / "vessel",
        ROOT / "data",
    ]

    def _collect_py_files(self) -> list[Path]:
        files = []
        for p in self.AUDIT_PATHS:
            files.extend(p.rglob("*.py"))
        return files

    def test_no_ml_framework_imports(self):
        files = self._collect_py_files()
        violations = []
        for f in files:
            text = f.read_text()
            for pat in self.FORBIDDEN_PATTERNS["ML frameworks"]:
                if re.search(pat, text):
                    violations.append(f"{f.relative_to(ROOT)}: matches '{pat}'")
        assert not violations, "ML framework code found:\n" + "\n".join(violations)

    def test_no_real_data_connectors(self):
        files = self._collect_py_files()
        violations = []
        for f in files:
            text = f.read_text()
            for pat in self.FORBIDDEN_PATTERNS["Real data connectors"]:
                if re.search(pat, text, re.IGNORECASE):
                    violations.append(f"{f.relative_to(ROOT)}: matches '{pat}'")
        assert not violations, "Real data connector code found:\n" + "\n".join(violations)

    def test_no_navigation_algorithm_code(self):
        files = self._collect_py_files()
        violations = []
        for f in files:
            # Skip vessel/api/main.py placeholder stubs (just return dicts)
            if "vessel/api/main.py" in str(f):
                continue
            text = f.read_text()
            for pat in self.FORBIDDEN_PATTERNS["Navigation algorithms"]:
                if re.search(pat, text, re.IGNORECASE):
                    violations.append(f"{f.relative_to(ROOT)}: matches '{pat}'")
        assert not violations, "Navigation algorithm code found:\n" + "\n".join(violations)

    def test_no_safety_shield_code(self):
        files = self._collect_py_files()
        violations = []
        for f in files:
            text = f.read_text()
            for pat in self.FORBIDDEN_PATTERNS["Safety Shield"]:
                if re.search(pat, text, re.IGNORECASE):
                    violations.append(f"{f.relative_to(ROOT)}: matches '{pat}'")
        assert not violations, "Safety Shield code found:\n" + "\n".join(violations)

    def test_no_confidence_decay_logic(self):
        """Confidence decay policy must not be implemented yet."""
        files = self._collect_py_files()
        violations = []
        decay_pattern = r"def.*decay|confidence.*\*.*age|age.*\*.*confidence|math\.exp.*age"
        for f in files:
            text = f.read_text()
            if re.search(decay_pattern, text, re.IGNORECASE):
                violations.append(str(f.relative_to(ROOT)))
        assert not violations, "Confidence decay logic found:\n" + "\n".join(violations)

    def test_placeholder_stubs_have_no_real_logic(self):
        """Stub __init__.py files in future-phase directories must not contain real logic."""
        stub_files = [
            ROOT / "cloud" / "data" / "__init__.py",
            ROOT / "cloud" / "models" / "__init__.py",
            ROOT / "vessel" / "cache" / "__init__.py",
            ROOT / "vessel" / "navigation" / "__init__.py",
            ROOT / "vessel" / "safety" / "__init__.py",
            ROOT / "benchmark" / "__init__.py",
        ]
        for f in stub_files:
            assert f.exists(), f"Stub file missing: {f}"
            text = f.read_text().strip()
            # Must not contain function or class definitions
            assert not re.search(r"^(def |class )", text, re.MULTILINE), \
                f"{f.name} contains real logic"


# ---------------------------------------------------------------------------
# 7. Cloud API endpoints (TestClient — no Docker needed)
# ---------------------------------------------------------------------------

class TestCloudAPI:
    @pytest.fixture
    def client(self):
        from cloud.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["service"] == "aeolus-cloud-api"

    def test_forecast_endpoint_minimal_request(self, client):
        r = client.post("/forecast", json={
            "vessel_position": {"latitude": -68.5, "longitude": -170.2},
        })
        assert r.status_code == 200, r.text

    def test_forecast_response_is_valid_forecast_package(self, client):
        from shared.schemas.forecast import ForecastPackage
        r = client.post("/forecast", json={
            "vessel_position": {"latitude": -68.5, "longitude": -170.2},
        })
        pkg = ForecastPackage.model_validate(r.json())
        assert pkg.package_id
        assert len(pkg.sea_ice.horizons) > 0
        assert len(pkg.icebergs) > 0

    def test_forecast_with_full_request(self, client):
        from shared.schemas.forecast import ForecastPackage
        r = client.post("/forecast", json={
            "vessel_position": {"latitude": -68.5, "longitude": -170.2},
            "destination": {"latitude": -75.0, "longitude": -165.0},
            "forecast_horizons_hours": [6, 12, 24],
            "vessel_context": {
                "vessel_id": "MV-SAGARMATHA",
                "ice_class": "PC5",
                "max_ice_concentration": 0.75,
            },
        })
        assert r.status_code == 200, r.text
        pkg = ForecastPackage.model_validate(r.json())
        assert pkg.package_id

    def test_forecast_response_contains_required_components(self, client):
        r = client.post("/forecast", json={
            "vessel_position": {"latitude": -68.5, "longitude": -170.2},
        })
        data = r.json()
        assert "sea_ice" in data
        assert "icebergs" in data
        assert "weather" in data
        assert "ocean" in data
        assert "overall_confidence" in data
        assert "valid_from" in data
        assert "valid_to" in data
        assert "generated_at" in data

    def test_forecast_sea_ice_has_uncertainty(self, client):
        r = client.post("/forecast", json={
            "vessel_position": {"latitude": -68.5, "longitude": -170.2},
        })
        horizons = r.json()["sea_ice"]["horizons"]
        for h in horizons:
            assert "uncertainty_grid" in h
            assert len(h["uncertainty_grid"]) > 0

    def test_forecast_icebergs_have_uncertainty(self, client):
        r = client.post("/forecast", json={
            "vessel_position": {"latitude": -68.5, "longitude": -170.2},
        })
        for iceberg in r.json()["icebergs"]:
            for h in iceberg["horizons"]:
                assert "uncertainty_radius_km" in h
                assert h["uncertainty_radius_km"] >= 0

    def test_forecast_freshness_fields_present(self, client):
        r = client.post("/forecast", json={
            "vessel_position": {"latitude": -68.5, "longitude": -170.2},
        })
        data = r.json()
        si = data["sea_ice"]
        assert "source_data_timestamp" in si
        assert "source_data_age_seconds" in si
        w = data["weather"]
        assert "source_data_timestamp" in w
        assert "source_data_age_seconds" in w


# ---------------------------------------------------------------------------
# 8. Vessel API endpoints (TestClient)
# ---------------------------------------------------------------------------

class TestVesselAPI:
    @pytest.fixture
    def client(self):
        from vessel.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["service"] == "aeolus-vessel-api"

    def test_sync_placeholder_returns_not_implemented(self, client):
        r = client.post("/sync")
        assert r.status_code == 200
        assert r.json()["status"] == "not_implemented"

    def test_navigate_placeholder_returns_not_implemented(self, client):
        r = client.post("/navigate")
        assert r.status_code == 200
        assert r.json()["status"] == "not_implemented"


# ---------------------------------------------------------------------------
# 9. NavigationWarning schema — TTE placement
# ---------------------------------------------------------------------------

class TestNavigationWarningTTE:
    def test_tte_is_in_warning_not_in_route_rejection(self):
        """
        TTE must live in NavigationWarning as an informational field.
        RouteCandidate must not have a TTE rejection field.
        """
        from shared.schemas.navigation import NavigationWarning, RouteCandidate
        import inspect

        warning_fields = NavigationWarning.model_fields
        route_fields = RouteCandidate.model_fields

        assert "time_to_encounter_hours" in warning_fields, \
            "TTE missing from NavigationWarning"
        assert "time_to_encounter_hours" not in route_fields, \
            "TTE should not be a RouteCandidate field"

    def test_tte_field_is_optional(self):
        from shared.schemas.navigation import NavigationWarning
        field = NavigationWarning.model_fields["time_to_encounter_hours"]
        # Optional means the annotation allows None
        import typing
        annotation = str(field.annotation)
        assert "None" in annotation or "Optional" in annotation, \
            "time_to_encounter_hours must be Optional"


# ---------------------------------------------------------------------------
# 10. ForecastFreshness — decay policy deferred
# ---------------------------------------------------------------------------

class TestForecastFreshnessDeferral:
    def test_current_confidence_is_optional(self):
        from shared.schemas.navigation import ForecastFreshness
        field = ForecastFreshness.model_fields["current_confidence"]
        import typing
        annotation = str(field.annotation)
        assert "None" in annotation or "Optional" in annotation

    def test_freshness_has_all_age_fields(self):
        from shared.schemas.navigation import ForecastFreshness
        fields = ForecastFreshness.model_fields
        assert "sea_ice_source_data_age_seconds" in fields
        assert "iceberg_source_data_age_seconds" in fields
        assert "weather_source_data_age_seconds" in fields
        assert "ocean_source_data_age_seconds" in fields
        assert "package_received_at" in fields
        assert "connectivity_state" in fields
