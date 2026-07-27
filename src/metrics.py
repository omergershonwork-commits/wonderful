"""Pure deterministic airport metric functions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.models import OperationalData, RouteRecord, TrafficRecord

MIN_PROJECTED_GROWTH = -0.10
MAX_PROJECTED_GROWTH = 0.20


@dataclass(frozen=True, slots=True)
class LongHaulMetrics:
    """Aggregated flight- and passenger-weighted long-haul results."""

    long_haul_departures: int
    all_departures: int
    departure_share: float | None
    long_haul_passengers: int
    all_route_passengers: int
    passenger_share: float | None
    qualifying_routes: tuple[RouteRecord, ...]


@dataclass(frozen=True, slots=True)
class ProjectedDemandMetrics:
    """Passenger projection after applying the approved growth clamp."""

    raw_growth: float
    clamped_growth: float
    projected_passengers: float


@dataclass(frozen=True, slots=True)
class UnmetCapacityMetrics:
    """Components of the estimated unmet-seat-capacity proxy."""

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


def passenger_growth(
    current_passengers: int,
    previous_passengers: int | None,
) -> float | None:
    """Return comparable-period passenger growth, or ``None`` for no denominator."""

    _require_non_negative("current_passengers", current_passengers)
    if previous_passengers is None:
        return None
    _require_non_negative("previous_passengers", previous_passengers)
    if previous_passengers == 0:
        return None
    return (current_passengers - previous_passengers) / previous_passengers


def load_factor(passengers: int, available_seats: int | None) -> float | None:
    """Return passengers divided by available seats."""

    _require_non_negative("passengers", passengers)
    if available_seats is None:
        return None
    _require_non_negative("available_seats", available_seats)
    if available_seats == 0:
        return None
    if passengers > available_seats:
        raise ValueError("passengers cannot exceed available_seats")
    return passengers / available_seats


def completion_rate(
    performed_departures: int,
    scheduled_departures: int,
) -> float | None:
    """Return performed departures divided by scheduled departures."""

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
    """Prefer reported cancellations, otherwise use ``1 - completion_rate``."""

    _require_non_negative("performed_departures", performed_departures)
    _require_non_negative("scheduled_departures", scheduled_departures)
    if scheduled_departures == 0:
        return None

    if reported_cancellations is not None:
        _require_non_negative("reported_cancellations", reported_cancellations)
        if reported_cancellations > scheduled_departures:
            raise ValueError(
                "reported_cancellations cannot exceed scheduled_departures"
            )
        return reported_cancellations / scheduled_departures

    completed = completion_rate(performed_departures, scheduled_departures)
    if completed is None:
        return None
    return max(0.0, min(1.0, 1.0 - completed))


def departures_per_runway(
    performed_departures: int,
    usable_runway_count: int | None,
) -> float | None:
    """Return the annual departures-per-usable-runway pressure proxy."""

    _require_non_negative("performed_departures", performed_departures)
    if usable_runway_count is None:
        return None
    if usable_runway_count <= 0:
        raise ValueError("usable_runway_count must be positive")
    return performed_departures / usable_runway_count


def long_haul_share(
    routes: Sequence[RouteRecord],
    threshold_miles: int = 3000,
) -> LongHaulMetrics:
    """Calculate flight- and passenger-weighted shares for qualifying routes."""

    if threshold_miles <= 0:
        raise ValueError("threshold_miles must be positive")

    all_departures = sum(route.departures for route in routes)
    all_passengers = sum(route.passengers for route in routes)
    qualifying = tuple(
        route for route in routes if route.distance_miles >= threshold_miles
    )
    long_haul_departures = sum(route.departures for route in qualifying)
    long_haul_passengers = sum(route.passengers for route in qualifying)

    return LongHaulMetrics(
        long_haul_departures=long_haul_departures,
        all_departures=all_departures,
        departure_share=(
            None if all_departures == 0 else long_haul_departures / all_departures
        ),
        long_haul_passengers=long_haul_passengers,
        all_route_passengers=all_passengers,
        passenger_share=(
            None if all_passengers == 0 else long_haul_passengers / all_passengers
        ),
        qualifying_routes=qualifying,
    )


def projected_demand(
    current_passengers: int,
    growth: float | None,
    minimum_growth: float = MIN_PROJECTED_GROWTH,
    maximum_growth: float = MAX_PROJECTED_GROWTH,
) -> ProjectedDemandMetrics | None:
    """Project passengers after clamping growth to the approved range."""

    _require_non_negative("current_passengers", current_passengers)
    if growth is None:
        return None
    if minimum_growth > maximum_growth:
        raise ValueError("minimum_growth cannot exceed maximum_growth")

    clamped_growth = max(minimum_growth, min(maximum_growth, growth))
    return ProjectedDemandMetrics(
        raw_growth=growth,
        clamped_growth=clamped_growth,
        projected_passengers=current_passengers * (1.0 + clamped_growth),
    )


def unmet_capacity_proxy(
    current_passengers: int,
    current_available_seats: int | None,
    growth: float | None,
    target_load_factor: float = 0.82,
) -> UnmetCapacityMetrics | None:
    """Return the approved estimated unmet-seat-capacity proxy."""

    if not 0 < target_load_factor <= 1:
        raise ValueError(
            "target_load_factor must be greater than 0 and at most 1"
        )
    _require_non_negative("current_passengers", current_passengers)
    if current_available_seats is None:
        return None
    _require_non_negative("current_available_seats", current_available_seats)

    projection = projected_demand(current_passengers, growth)
    if projection is None:
        return None

    required_seats = projection.projected_passengers / target_load_factor
    unmet_capacity = max(0.0, required_seats - current_available_seats)
    return UnmetCapacityMetrics(
        current_passengers=current_passengers,
        current_available_seats=current_available_seats,
        raw_growth=projection.raw_growth,
        clamped_growth=projection.clamped_growth,
        projected_passengers=projection.projected_passengers,
        target_load_factor=target_load_factor,
        required_seats=required_seats,
        estimated_unmet_capacity_proxy=unmet_capacity,
    )


def metrics_from_records(
    traffic: TrafficRecord,
    operations: OperationalData,
    *,
    target_load_factor: float = 0.82,
) -> dict[str, float | None]:
    """Calculate all raw airport metrics used by scoring and tools."""

    growth = passenger_growth(
        traffic.passengers,
        traffic.previous_period_passengers,
    )
    capacity = unmet_capacity_proxy(
        traffic.passengers,
        traffic.available_seats,
        growth,
        target_load_factor,
    )
    return {
        "passenger_growth": growth,
        "load_factor": load_factor(
            traffic.passengers,
            traffic.available_seats,
        ),
        "completion_rate": completion_rate(
            operations.performed_departures,
            operations.scheduled_departures,
        ),
        "cancellation_rate": cancellation_rate(
            operations.reported_cancellations,
            operations.performed_departures,
            operations.scheduled_departures,
        ),
        "departures_per_runway": departures_per_runway(
            operations.performed_departures,
            operations.usable_runway_count,
        ),
        "estimated_unmet_capacity_proxy": (
            None if capacity is None else capacity.estimated_unmet_capacity_proxy
        ),
        "market_scale": float(traffic.passengers),
        "average_departure_delay_minutes": (
            operations.average_departure_delay_minutes
        ),
        "average_taxi_out_minutes": operations.average_taxi_out_minutes,
    }
