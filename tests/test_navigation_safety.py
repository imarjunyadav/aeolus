from datetime import datetime, timezone
from data.mock.generator import generate_mock_forecast_package
from shared.schemas.common import Position
from shared.schemas.vessel import VesselContext, VesselState
from vessel.navigation.planner import plan_navigation


def test_navigation_returns_alternatives_when_safe():
    package = generate_mock_forecast_package(Position(latitude=-68.5, longitude=-170.2), seed=7, reference_time=datetime(2026, 9, 17, tzinfo=timezone.utc))
    state = VesselState(vessel_id="T", timestamp=datetime.now(timezone.utc), latitude=-68.5, longitude=-170.2, speed_knots=10, heading_deg=90, destination=Position(latitude=-66, longitude=-164), context=VesselContext(vessel_id="T", max_ice_concentration=1.0, min_iceberg_clearance_km=0))
    result = plan_navigation(package, state)
    assert result.recommended_route is not None
    assert all(r.safety_status.value == "APPROVED" for r in result.alternative_routes)


def test_no_safe_route_when_destination_exceeds_ice_limit():
    package = generate_mock_forecast_package(Position(latitude=-85, longitude=0), seed=1, reference_time=datetime(2026, 9, 17, tzinfo=timezone.utc))
    state = VesselState(vessel_id="T", timestamp=datetime.now(timezone.utc), latitude=-85, longitude=0, speed_knots=10, heading_deg=0, destination=Position(latitude=-89, longitude=0), context=VesselContext(vessel_id="T", max_ice_concentration=0.0, min_iceberg_clearance_km=0))
    result = plan_navigation(package, state)
    assert result.no_safe_route
