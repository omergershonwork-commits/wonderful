"""Tests for deterministic airport metrics."""

from datetime import date, datetime, timezone

import pytest

from src.metrics import (
    cancellation_rate,
    completion_rate,
    departures_per_runway,
    load_factor,
    long_haul_share,
    passenger_growth,
    unmet_capacity_proxy,
)
from src.models import DataMode, DataPeriod, RouteRecord, SourceMetadata


def _route(
    distance_miles: int,
    departures: int,
    passengers: int,
    destination_code: str,
) -> RouteRecord:
    period = DataPeriod(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
    )
    source = SourceMetadata(
        source_name="Metric test fixture",
        data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
        retrieved_at=datetime.now(timezone.utc),
        period=period,
    )
    return RouteRecord(
        origin_airport_code="ANC",
        destination_airport_code=destination_code,
        distance_miles=distance_miles,
        departures=departures,
        passengers=passengers,
        period=period,
        source=source,
    )


def test_basic_metric_formulas() -> None:
    assert passenger_growth(120, 100) == 0.2
    assert load_factor(80, 100) == 0.8
    assert completion_rate(90, 100) == 0.9
    assert cancellation_rate(7, 90, 100) == 0.07
    assert cancellation_rate(None, 90, 100) == pytest.approx(0.1)
    assert departures_per_runway(100, 4) == 25


def test_zero_denominators_return_unavailable() -> None:
    assert passenger_growth(10, 0) is None
    assert load_factor(0, 0) is None
    assert completion_rate(0, 0) is None
    assert cancellation_rate(None, 0, 0) is None


def test_exact_threshold_is_included_and_weightings_differ() -> None:
    result = long_haul_share(
        [
            _route(2999, 90, 900, "SEA"),
            _route(3000, 10, 500, "DFW"),
        ]
    )

    assert result.long_haul_departures == 10
    assert result.departure_share == pytest.approx(0.1)
    assert result.passenger_share == pytest.approx(500 / 1400)


def test_growth_cap_and_nonnegative_unmet_capacity() -> None:
    high_growth = unmet_capacity_proxy(1000, 1000, 0.9, 0.8)
    low_growth = unmet_capacity_proxy(1000, 2000, -0.9, 0.8)

    assert high_growth is not None
    assert high_growth.clamped_growth == 0.2
    assert high_growth.projected_passengers == 1200
    assert low_growth is not None
    assert low_growth.clamped_growth == -0.1
    assert low_growth.estimated_unmet_capacity_proxy == 0


def test_target_load_factor_is_validated() -> None:
    with pytest.raises(ValueError, match="target_load_factor"):
        unmet_capacity_proxy(100, 100, 0.1, 0)

    with pytest.raises(ValueError, match="target_load_factor"):
        unmet_capacity_proxy(100, 100, 0.1, 1.01)
