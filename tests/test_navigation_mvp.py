from datetime import datetime, timezone

from data.mock.generator import generate_mock_forecast_package
from shared.schemas.common import OptimizationMode, Position
from shared.schemas.vessel import VesselContext, VesselState
from vessel.navigation.planner import plan_navigation


def test_navigation_produces_safe_route_from_mock_package():
    package = generate_mock_forecast_package(Position(latitude=-68.5, longitude=-170.2), seed=7, reference_time=datetime(2026, 9, 17, tzinfo=timezone.utc))
    state = VesselState(
        vessel_id="TEST-1",
        timestamp=datetime(2026, 9, 17, tzinfo=timezone.utc),
        latitude=-68.5,
        longitude=-170.2,
        speed_knots=10,
        heading_deg=90,
        destination=Position(latitude=-66.0, longitude=-164.0),
        context=VesselContext(vessel_id="TEST-1", max_ice_concentration=1.0, min_iceberg_clearance_km=0),
    )
    result = plan_navigation(package, state, OptimizationMode.RISK)
    assert result.forecast_package_id == package.package_id
    assert result.recommended_route is not None or result.no_safe_route
    if result.recommended_route:
        assert result.recommended_route.safety_status.value == "APPROVED"
