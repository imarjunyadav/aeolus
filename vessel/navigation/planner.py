"""Deterministic safety-first navigation planner for Aeolus.

This is the Phase-5/6 engineering baseline.  It is intentionally not ML or
LLM driven: environmental forecasts become a cost/risk grid, hard safety
constraints are applied first, and A* searches only feasible cells.
"""
from __future__ import annotations

import heapq
import math
import os
import uuid
from datetime import datetime, timezone
from typing import Iterable

from shared.schemas.common import ConnectivityState, OptimizationMode, Position, RiskLevel, SafetyStatus, WarnCategory, WarnLevel
from shared.schemas.forecast import ForecastPackage, IcebergForecast
from shared.schemas.navigation import ForecastFreshness, NavigationResult, NavigationWarning, RouteCandidate, RouteWaypoint
from shared.schemas.vessel import VesselContext, VesselState


def haversine_nm(a: Position, b: Position) -> float:
    r_km = 6371.0088
    p1, p2 = math.radians(a.latitude), math.radians(b.latitude)
    dp = math.radians(b.latitude - a.latitude)
    dl = math.radians(b.longitude - a.longitude)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return (2 * r_km * math.asin(min(1.0, math.sqrt(h)))) / 1.852


def _risk_level(score: float) -> RiskLevel:
    if score >= 0.8:
        return RiskLevel.EXTREME
    if score >= 0.6:
        return RiskLevel.HIGH
    if score >= 0.3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _nearest_grid_index(value: float, lo: float, hi: float, n: int) -> int:
    if n <= 1 or hi == lo:
        return 0
    return max(0, min(n - 1, int(round((value - lo) / (hi - lo) * (n - 1)))))


def _grid_position(package: ForecastPackage, i: int, j: int) -> Position:
    g = package.sea_ice.grid_metadata
    return Position(
        latitude=g.lat_min + (g.lat_max - g.lat_min) * i / max(g.n_lat - 1, 1),
        longitude=g.lon_min + (g.lon_max - g.lon_min) * j / max(g.n_lon - 1, 1),
    )


def _nearest_iceberg_distance_km(point: Position, icebergs: Iterable[IcebergForecast], horizon_hours: int) -> float:
    distances = []
    for iceberg in icebergs:
        if iceberg.horizons:
            h = min(iceberg.horizons, key=lambda x: abs(x.horizon_hours - horizon_hours))
            target = Position(latitude=h.latitude, longitude=h.longitude)
        else:
            target = iceberg.last_known_position
        distances.append(haversine_nm(point, target) * 1.852)
    return min(distances) if distances else float("inf")


def _iceberg_risk(distance_km: float) -> float:
    if math.isinf(distance_km):
        return 0.0
    return max(0.0, min(1.0, math.exp(-distance_km / 8.0)))


def _freshness(package: ForecastPackage, now: datetime) -> ForecastFreshness:
    ages = [package.sea_ice.source_data_age_seconds, package.weather.source_data_age_seconds, package.ocean.source_data_age_seconds]
    ages.extend(x.source_data_age_seconds for x in package.icebergs)
    max_age = max(ages, default=0)
    stale_limit = int(float(os.getenv("AEOLUS_STALE_DATA_HOURS", "72")) * 3600)
    if max_age > stale_limit:
        state = ConnectivityState.OFFLINE
    elif max_age > stale_limit / 2:
        state = ConnectivityState.DEGRADED
    else:
        state = ConnectivityState.LIVE
    # Phase-1 schema deliberately does not define a confidence adjustment formula.
    # Navigation reports the cloud confidence unchanged; freshness/connectivity
    # are separate operational signals until real forecast-error analysis defines a policy.
    return ForecastFreshness(
        package_id=package.package_id,
        package_received_at=now,
        sea_ice_source_data_age_seconds=package.sea_ice.source_data_age_seconds,
        iceberg_source_data_age_seconds=max((x.source_data_age_seconds for x in package.icebergs), default=0),
        weather_source_data_age_seconds=package.weather.source_data_age_seconds,
        ocean_source_data_age_seconds=package.ocean.source_data_age_seconds,
        current_confidence=package.overall_confidence,
        connectivity_state=state,
    )


def _build_warnings(package: ForecastPackage, state: VesselState, now: datetime) -> list[NavigationWarning]:
    warnings: list[NavigationWarning] = []
    vessel = Position(latitude=state.latitude, longitude=state.longitude)
    for iceberg in package.icebergs:
        if not iceberg.horizons:
            continue
        # Evaluate the earliest predicted point as an alert; route safety uses
        # the horizon closest to each route-cell ETA separately.
        h = min(iceberg.horizons, key=lambda x: x.horizon_hours)
        distance = haversine_nm(vessel, Position(latitude=h.latitude, longitude=h.longitude)) * 1.852
        if distance < 30:
            speed_kmh = max(state.speed_knots * 1.852, 1.0)
            tte = distance / speed_kmh
            warnings.append(NavigationWarning(
                warning_id=f"WARN-{uuid.uuid4().hex[:10]}",
                level=WarnLevel.DANGER if distance < 5 else WarnLevel.WARNING if distance < 10 else WarnLevel.CAUTION,
                category=WarnCategory.ICEBERG_ENCOUNTER,
                message=f"Predicted iceberg {iceberg.iceberg_id} is {distance:.1f} km from current vessel position.",
                iceberg_id=iceberg.iceberg_id,
                time_to_encounter_hours=tte,
                estimated_clearance_km=distance,
                issued_at=now,
            ))
    return warnings


def _route_for_mode(package: ForecastPackage, state: VesselState, context: VesselContext, mode: OptimizationMode):
    destination = state.destination
    if destination is None:
        raise ValueError("VesselState.destination is required for navigation")
    grid = package.sea_ice.grid_metadata
    if grid.n_lat < 1 or grid.n_lon < 1:
        raise ValueError("ForecastPackage grid is empty")
    horizon = package.sea_ice.horizons[0]
    start = (_nearest_grid_index(state.latitude, grid.lat_min, grid.lat_max, grid.n_lat), _nearest_grid_index(state.longitude, grid.lon_min, grid.lon_max, grid.n_lon))
    goal = (_nearest_grid_index(destination.latitude, grid.lat_min, grid.lat_max, grid.n_lat), _nearest_grid_index(destination.longitude, grid.lon_min, grid.lon_max, grid.n_lon))
    max_ice = context.max_ice_concentration if context.max_ice_concentration is not None else 1.0
    min_clearance = context.min_iceberg_clearance_km if context.min_iceberg_clearance_km is not None else 0.0

    def cell(node):
        i, j = node
        conc = float(horizon.concentration_grid[i][j])
        pos = _grid_position(package, i, j)
        iceberg_dist = _nearest_iceberg_distance_km(pos, package.icebergs, horizon.horizon_hours)
        iceberg_risk = _iceberg_risk(iceberg_dist)
        # Provider snapshots are area-representative, so weather/ocean are
        # intentionally small modifiers rather than pretending to be gridded.
        weather_risk = min(1.0, (package.weather.wind_speed_ms or 0.0) / 35.0)
        wave_risk = min(1.0, (package.weather.significant_wave_height_m or 0.0) / 8.0)
        current = math.hypot(package.ocean.current_u_ms or 0.0, package.ocean.current_v_ms or 0.0)
        ocean_risk = min(1.0, current / 1.5)
        environmental = min(1.0, 0.55 * conc + 0.25 * iceberg_risk + 0.12 * weather_risk + 0.05 * wave_risk + 0.03 * ocean_risk)
        reasons = []
        if conc > max_ice:
            reasons.append(f"ice concentration {conc:.2f} exceeds vessel limit {max_ice:.2f}")
        if iceberg_dist < min_clearance:
            reasons.append(f"iceberg clearance {iceberg_dist:.1f} km below minimum {min_clearance:.1f} km")
        return pos, conc, iceberg_dist, environmental, reasons

    if cell(start)[4] or cell(goal)[4]:
        return None, [cell(start)[4], cell(goal)[4]]

    risk_weight = {OptimizationMode.RISK: 8.0, OptimizationMode.BALANCED: 3.0, OptimizationMode.FUEL: 1.0, OptimizationMode.SHORTEST: 0.1}[mode]
    neighbors = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]

    def heuristic(node):
        return haversine_nm(_grid_position(package, *node), destination)

    def edge_cost(a, b):
        pa = cell(a)[0]
        environmental = cell(b)[3]
        distance = haversine_nm(pa, cell(b)[0])
        return distance * (1.0 + risk_weight * environmental)

    frontier = [(heuristic(start), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal:
            break
        for di, dj in neighbors:
            nxt = (current[0] + di, current[1] + dj)
            if not (0 <= nxt[0] < grid.n_lat and 0 <= nxt[1] < grid.n_lon) or cell(nxt)[4]:
                continue
            new_cost = current_cost + edge_cost(current, nxt)
            if new_cost < g_score.get(nxt, float("inf")):
                g_score[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(frontier, (new_cost + heuristic(nxt), new_cost, nxt))

    if goal not in came_from and goal != start:
        return None, ["No grid path satisfies the hard safety constraints."]

    nodes = [goal]
    while nodes[-1] != start:
        nodes.append(came_from[nodes[-1]])
    nodes.reverse()
    points = [_grid_position(package, *node) for node in nodes]
    total_distance = sum(haversine_nm(points[i], points[i + 1]) for i in range(len(points) - 1))
    route_waypoints = []
    risk_values = []
    for node, point in zip(nodes, points):
        _, conc, _, risk, _ = cell(node)
        risk_values.append(risk)
        route_waypoints.append(RouteWaypoint(position=point, ice_concentration_at_point=conc, risk_score_at_point=risk))
    risk_score = sum(risk_values) / len(risk_values)
    speed = max(state.speed_knots, 0.5)
    eta = total_distance / speed
    fuel = eta * (1.0 + risk_score)
    return RouteCandidate(
        route_id=f"ROUTE-{uuid.uuid4().hex[:10]}",
        optimization_mode=mode,
        waypoints=route_waypoints,
        total_distance_nm=total_distance,
        estimated_fuel_relative=fuel,
        estimated_eta_hours=eta,
        risk_score=risk_score,
        risk_level=_risk_level(risk_score),
        safety_status=SafetyStatus.APPROVED,
    ), []


def plan_navigation(package: ForecastPackage, state: VesselState, mode: OptimizationMode = OptimizationMode.BALANCED) -> NavigationResult:
    now = datetime.now(tz=timezone.utc)
    destination = state.destination
    if destination is None:
        raise ValueError("VesselState.destination is required for navigation")
    context = state.context or VesselContext(vessel_id=state.vessel_id)
    freshness = _freshness(package, now)
    warnings = _build_warnings(package, state, now)

    if freshness.connectivity_state == ConnectivityState.OFFLINE and freshness.current_confidence is not None and freshness.current_confidence < 0.10:
        return NavigationResult(result_id=f"NAV-{uuid.uuid4().hex[:10]}", computed_at=now, vessel_state_timestamp=state.timestamp, forecast_package_id=package.package_id, recommended_route=None, alternative_routes=[], rejected_routes=[], no_safe_route=True, no_safe_route_reason="Environmental forecast is too stale to support a safe recommendation.", active_warnings=warnings, forecast_freshness=freshness, connectivity_state=freshness.connectivity_state)

    # Generate all four deterministic optimization views. Safety is evaluated
    # inside _route_for_mode before any candidate can be ranked.
    candidates: list[RouteCandidate] = []
    rejected: list[RouteCandidate] = []
    for candidate_mode in OptimizationMode:
        candidate, reasons = _route_for_mode(package, state, context, candidate_mode)
        if candidate is not None:
            candidates.append(candidate)
        else:
            rejected.append(RouteCandidate(
                route_id=f"REJECTED-{uuid.uuid4().hex[:10]}",
                optimization_mode=candidate_mode,
                waypoints=[], total_distance_nm=0.0,
                safety_status=SafetyStatus.REJECTED,
                rejection_reasons=[str(r) for r in reasons if r],
            ))

    if not candidates:
        warnings.append(NavigationWarning(
            warning_id=f"WARN-{uuid.uuid4().hex[:10]}", level=WarnLevel.DANGER,
            category=WarnCategory.NO_SAFE_ROUTE,
            message="No candidate route satisfies the configured hard safety constraints.", issued_at=now,
        ))
        return NavigationResult(result_id=f"NAV-{uuid.uuid4().hex[:10]}", computed_at=now, vessel_state_timestamp=state.timestamp, forecast_package_id=package.package_id, no_safe_route=True, no_safe_route_reason="All candidate routes were rejected by the Safety Shield.", active_warnings=warnings, forecast_freshness=freshness, connectivity_state=freshness.connectivity_state, rejected_routes=rejected)

    def selection_key(route: RouteCandidate):
        if mode == OptimizationMode.RISK:
            return (route.risk_score or 1.0, route.total_distance_nm)
        if mode == OptimizationMode.FUEL:
            return (route.estimated_fuel_relative or float("inf"), route.risk_score or 1.0)
        if mode == OptimizationMode.SHORTEST:
            return (route.total_distance_nm, route.risk_score or 1.0)
        return ((route.risk_score or 1.0) * 3.0 + (route.estimated_eta_hours or 0.0) / 100.0, route.total_distance_nm)

    recommended = min(candidates, key=selection_key)
    alternatives = [r for r in sorted(candidates, key=selection_key) if r.route_id != recommended.route_id][:3]
    return NavigationResult(result_id=f"NAV-{uuid.uuid4().hex[:10]}", computed_at=now, vessel_state_timestamp=state.timestamp, forecast_package_id=package.package_id, recommended_route=recommended, alternative_routes=alternatives, rejected_routes=rejected, no_safe_route=False, active_warnings=warnings, forecast_freshness=freshness, connectivity_state=freshness.connectivity_state)
