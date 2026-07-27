"""Pure deterministic airport metric functions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.models import OperationalData, RouteRecord, TrafficRecord

MIN_PROJECTED_GROWTH = -0.10
MAX_PROJECTED_GROWTH = 0.20


@dataclass(frozen=True, slots=True)
class LongHaulMetrics:
    long_haul_departures: int
    all_departures: int
    departure_share: float | None
    long_haul_passengers: int
    all_route_passengers: int
    passenger_share: float | None
    qualifying_routes: tuple[RouteRecord, ...]


@dataclass(frozen=True, slots=True)
class ProjectedDemandMetrics:
    raw_growth: float
    clamped_growth: float
    projected_passengers: float


@dataclass(frozen=True, slots=True)
class UnmetCapacityMetrics:
    current_passengers: int
    current_available_seats: int
    raw_growth: float
    clamped_growth: float
    projected_passengers: float
    target_load_factor: float
    required_seats: float
    estimated_unmet_capacity_proxy: float


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def passenger_growth(current_passengers: int, previous_passengers: int | None) -> float | None:
    _require_non_negative("current_passengers", current_passengers)
    if previous_passengers is None:
        return None
    _require_non_negative("previous_passengers", previous_passengers)
    if previous_passengers == 0:
        return None
    return (current_passengers - previous_passengers) / previous_passengers


def load_factor(passengers: int, available_seats: int | None) -> float | None:
    _require_non_negative("passengers", passengers)
    if available_seats is None:
        return None
    _require_non_negative("available_seats", available_seats)
    if available_seats == 0:
        return None
    if passengers > available_seats:
        raise ValueError("passengers cannot exceed available_seats")
    return passengers / available_seats


def completion_rate(performed_departures: int, scheduled_departures: int) -> float | None:
    _require_non_negative("performed_departures", performed_departures)
    _require_non_negative("scheduled_departures", scheduled_departures)
    if scheduled_departures == 0:
        return None
    if performed_departures > scheduled_departures:
        raise ValueError("performed_departures cannot exceed scheduled_departures")
    return performed_departures / scheduled_departures


def cancellation_rate(
    reported_cancellations: int | None,
    performed_departures: int,
    scheduled_departures: int,
) -> float | None:
    _require_non_negative("performed_departures", performed_departures)
    _require_non_negative("scheduled_departures", scheduled_departures)
    if performed_departures > scheduled_departures:
        raise ValueError("performed_departures cannot exceed scheduled_departures")
    if scheduled_departures == 0:
        if reported_cancellations not in {None, 0}:
            raise ValueError("reported cancellations contradict departure totals")
        return None
    expected = scheduled_departures - performed_departures
    if reported_cancellations is not None:
        _require_non_negative("reported_cancellations", reported_cancellations)
        if reported_cancellations != expected:
            raise ValueError("reported cancellations contradict departure totals")
        return reported_cancellations / scheduled_departures
    return expected / scheduled_departures


def departures_per_runway(performed_departures: int, usable_runway_count: int | None) -> float | None:
    _require_non_negative("performed_departures", performed_departures)
    if usable_runway_count is None:
        return None
    if usable_runway_count <= 0:
        raise ValueError("usable_runway_count must be positive")
    return performed_departures / usable_runway_count


def long_haul_share(routes: Sequence[RouteRecord], threshold_miles: int = 3000) -> LongHaulMetrics:
    if threshold_miles <= 0:
        raise ValueError("threshold_miles must be positive")
    if routes:
        origins = {route.origin_airport_code for route in routes}
        periods = {(route.period.start_date, route.period.end_date) for route in routes}
        if len(origins) != 1:
            raise ValueError("routes must share one origin airport")
        if len(periods) != 1:
            raise ValueError("routes must share one analysis period")
    all_departures = sum(route.departures for route in routes)
    all_passengers = sum(route.passengers for route in routes)
    qualifying = tuple(route for route in routes if route.distance_miles >= threshold_miles)
    long_departures = sum(route.departures for route in qualifying)
    long_passengers = sum(route.passengers for route in qualifying)
    return LongHaulMetrics(
        long_haul_departures=long_departures,
        all_departures=all_departures,
        departure_share=None if all_departures == 0 else long_departures / all_departures,
        long_haul_passengers=long_passengers,
        all_route_passengers=all_passengers,
        passenger_share=None if all_passengers == 0 else long_passengers / all_passengers,
        qualifying_routes=qualifying,
    )


def projected_demand(
    current_passengers: int,
    growth: float | None,
    minimum_growth: float = MIN_PROJECTED_GROWTH,
    maximum_growth: float = MAX_PROJECTED_GROWTH,
) -> ProjectedDemandMetrics | None:
    _require_non_negative("current_passengers", current_passengers)
    if growth is None:
        return None
    if minimum_growth > maximum_growth:
        raise ValueError("minimum_growth cannot exceed maximum_growth")
    clamped = max(minimum_growth, min(maximum_growth, growth))
    return ProjectedDemandMetrics(growth, clamped, current_passengers * (1 + clamped))


def unmet_capacity_proxy(
    current_passengers: int,
    current_available_seats: int | None,
    growth: float | None,
    target_load_factor: float = 0.82,
) -> UnmetCapacityMetrics | None:
    if not 0 < target_load_factor <= 1:
        raise ValueError("target_load_factor must be greater than 0 and at most 1")
    _require_non_negative("current_passengers", current_passengers)
    if current_available_seats is None:
        return None
    _require_non_negative("current_available_seats", current_available_seats)
    if current_passengers > current_available_seats:
        raise ValueError("current_passengers cannot exceed current_available_seats")
    projection = projected_demand(current_passengers, growth)
    if projection is None:
        return None
    required = projection.projected_passengers / target_load_factor
    unmet = max(0.0, required - current_available_seats)
    return UnmetCapacityMetrics(
        current_passengers=current_passengers,
        current_available_seats=current_available_seats,
        raw_growth=projection.raw_growth,
        clamped_growth=projection.clamped_growth,
        projected_passengers=projection.projected_passengers,
        target_load_factor=target_load_factor,
        required_seats=required,
        estimated_unmet_capacity_proxy=unmet,
    )


def metrics_from_records(
    traffic: TrafficRecord,
    operations: OperationalData,
    *,
    target_load_factor: float = 0.82,
) -> dict[str, float | None]:
    if traffic.airport_code != operations.airport_code:
        raise ValueError("traffic and operations must describe the same airport")
    if traffic.period != operations.period:
        raise ValueError("traffic and operations must use the same analysis period")
    growth = passenger_growth(traffic.passengers, traffic.previous_period_passengers)
    capacity = unmet_capacity_proxy(
        traffic.passengers,
        traffic.available_seats,
        growth,
        target_load_factor,
    )
    return {
        "passenger_growth": growth,
        "load_factor": load_factor(traffic.passengers, traffic.available_seats),
        "completion_rate": completion_rate(
            operations.performed_departures, operations.scheduled_departures
        ),
        "cancellation_rate": cancellation_rate(
            operations.reported_cancellations,
            operations.performed_departures,
            operations.scheduled_departures,
        ),
        "departures_per_runway": departures_per_runway(
            operations.performed_departures, operations.usable_runway_count
        ),
        "estimated_unmet_capacity_proxy": (
            None if capacity is None else capacity.estimated_unmet_capacity_proxy
        ),
        "market_scale": float(traffic.passengers),
        "average_departure_delay_minutes": operations.average_departure_delay_minutes,
        "average_taxi_out_minutes": operations.average_taxi_out_minutes,
    }
