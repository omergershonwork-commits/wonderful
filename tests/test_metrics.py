"""Tests for deterministic airport metrics."""

from datetime import date, datetime, timezone

import pytest

from src.metrics import (
    cancellation_rate,
    completion_rate,
    departures_per_runway,
    load_factor,
    long_haul_share,
    metrics_from_records,
    passenger_growth,
    unmet_capacity_proxy,
)
from src.models import (
    DataMode,
    DataPeriod,
    OperationalData,
    RouteRecord,
    SourceMetadata,
    TrafficRecord,
)


def _period(year: int = 2025) -> DataPeriod:
    return DataPeriod(start_date=date(year, 1, 1), end_date=date(year, 12, 31))


def _source(period: DataPeriod) -> SourceMetadata:
    return SourceMetadata(
        source_name="Metric test fixture",
        data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
        retrieved_at=datetime.now(timezone.utc),
        period=period,
    )


def _route(
    distance_miles: int,
    departures: int,
    passengers: int,
    destination_code: str,
    *,
    origin: str = "ANC",
    period: DataPeriod | None = None,
) -> RouteRecord:
    period = period or _period()
    return RouteRecord(
        origin_airport_code=origin,
        destination_airport_code=destination_code,
        distance_miles=distance_miles,
        departures=departures,
        passengers=passengers,
        period=period,
        source=_source(period),
    )


def _traffic(code: str = "SFO", period: DataPeriod | None = None) -> TrafficRecord:
    period = period or _period()
    previous = _period(period.start_date.year - 1)
    return TrafficRecord(
        airport_code=code,
        period=period,
        passengers=100,
        previous_period_passengers=90,
        previous_period=previous,
        previous_source=_source(previous),
        available_seats=120,
        source=_source(period),
    )


def _operations(code: str = "SFO", period: DataPeriod | None = None) -> OperationalData:
    period = period or _period()
    return OperationalData(
        airport_code=code,
        period=period,
        scheduled_departures=100,
        performed_departures=90,
        reported_cancellations=10,
        average_departure_delay_minutes=10,
        average_taxi_out_minutes=12,
        usable_runway_count=2,
        source=_source(period),
    )


def test_basic_metric_formulas() -> None:
    assert passenger_growth(120, 100) == 0.2
    assert load_factor(80, 100) == 0.8
    assert completion_rate(90, 100) == 0.9
    assert cancellation_rate(10, 90, 100) == 0.1
    assert cancellation_rate(None, 90, 100) == pytest.approx(0.1)
    assert departures_per_runway(100, 4) == 25


def test_zero_denominators_return_unavailable() -> None:
    assert passenger_growth(10, 0) is None
    assert load_factor(0, 0) is None
    assert completion_rate(0, 0) is None
    assert cancellation_rate(None, 0, 0) is None


def test_exact_threshold_is_included_and_weightings_differ() -> None:
    result = long_haul_share(
        [_route(2999, 90, 900, "SEA"), _route(3000, 10, 500, "DFW")]
    )
    assert result.long_haul_departures == 10
    assert result.departure_share == pytest.approx(0.1)
    assert result.passenger_share == pytest.approx(500 / 1400)


def test_growth_cap_and_nonnegative_unmet_capacity() -> None:
    high_growth = unmet_capacity_proxy(1000, 1000, 0.9, 0.8)
    low_growth = unmet_capacity_proxy(1000, 2000, -0.9, 0.8)
    assert high_growth is not None and high_growth.clamped_growth == 0.2
    assert high_growth.projected_passengers == 1200
    assert low_growth is not None and low_growth.clamped_growth == -0.1
    assert low_growth.estimated_unmet_capacity_proxy == 0


def test_target_load_factor_and_passenger_seat_integrity_are_validated() -> None:
    with pytest.raises(ValueError, match="target_load_factor"):
        unmet_capacity_proxy(100, 100, 0.1, 0)
    with pytest.raises(ValueError, match="target_load_factor"):
        unmet_capacity_proxy(100, 100, 0.1, 1.01)
    with pytest.raises(ValueError, match="cannot exceed"):
        unmet_capacity_proxy(101, 100, 0.1, 0.82)


def test_metrics_from_records_rejects_cross_airport_and_cross_period_inputs() -> None:
    with pytest.raises(ValueError, match="same airport"):
        metrics_from_records(_traffic("SFO"), _operations("LAX"))
    with pytest.raises(ValueError, match="same analysis period"):
        metrics_from_records(_traffic(period=_period(2025)), _operations(period=_period(2024)))


def test_cancellation_contradictions_are_rejected() -> None:
    with pytest.raises(ValueError, match="contradict"):
        cancellation_rate(5, 90, 100)


def test_long_haul_rejects_mixed_origins_and_periods() -> None:
    with pytest.raises(ValueError, match="one origin"):
        long_haul_share([
            _route(3000, 1, 1, "DFW", origin="ANC"),
            _route(3001, 1, 1, "JFK", origin="SFO"),
        ])
    with pytest.raises(ValueError, match="one analysis period"):
        long_haul_share([
            _route(3000, 1, 1, "DFW", period=_period(2025)),
            _route(3001, 1, 1, "JFK", period=_period(2024)),
        ])
