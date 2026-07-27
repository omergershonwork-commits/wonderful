"""Tests for normalized domain models and approved tool boundaries."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.models import (
    AirportRecord,
    CalculateLongHaulShareInput,
    CalculatedMetrics,
    CompareAirportsInput,
    ConfidenceInfo,
    ConfidenceLevel,
    DataMode,
    DataPeriod,
    EstimateUnmetCapacityInput,
    OperationalData,
    RankAirportsInput,
    RouteRecord,
    SourceMetadata,
    TrafficRecord,
    UnavailableDataResponse,
)


@pytest.fixture
def analysis_period() -> DataPeriod:
    return DataPeriod(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        label="Calendar year 2025",
    )


@pytest.fixture
def demo_source(analysis_period: DataPeriod) -> SourceMetadata:
    return SourceMetadata(
        source_name="Illustrative Phase 3 fixture",
        data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
        period=analysis_period,
        notes=["Values exist only to validate the normalized schema."],
    )


def test_sfo_fixture_validates(
    analysis_period: DataPeriod,
    demo_source: SourceMetadata,
) -> None:
    airport = AirportRecord(
        airport_code="sfo",
        name="San Francisco International Airport",
        city="San Francisco",
        state_code="ca",
        region="West",
        latitude=37.6213,
        longitude=-122.3790,
        usable_runway_count=4,
        source=demo_source,
    )
    traffic = TrafficRecord(
        airport_code="SFO",
        period=analysis_period,
        passengers=52_000_000,
        previous_period_passengers=49_000_000,
        available_seats=60_000_000,
        source=demo_source,
    )
    operations = OperationalData(
        airport_code="SFO",
        period=analysis_period,
        scheduled_departures=190_000,
        performed_departures=182_000,
        reported_cancellations=8_000,
        average_departure_delay_minutes=17.5,
        average_taxi_out_minutes=19.2,
        usable_runway_count=4,
        source=demo_source,
    )
    route = RouteRecord(
        origin_airport_code="sfo",
        destination_airport_code="jfk",
        destination_name="John F. Kennedy International Airport",
        distance_miles=2586,
        departures=4000,
        passengers=650_000,
        available_seats=720_000,
        period=analysis_period,
        source=demo_source,
    )

    assert airport.airport_code == "SFO"
    assert airport.state_code == "CA"
    assert traffic.passengers == 52_000_000
    assert operations.performed_departures == 182_000
    assert route.destination_airport_code == "JFK"
    assert airport.model_dump(mode="json")["source"]["data_mode"] == (
        "ILLUSTRATIVE DEMO DATA"
    )


def test_invalid_airport_code_is_rejected(demo_source: SourceMetadata) -> None:
    with pytest.raises(ValidationError, match="exactly three letters"):
        AirportRecord(
            airport_code="SF",
            name="Invalid Airport",
            city="San Francisco",
            state_code="CA",
            source=demo_source,
        )


def test_data_period_rejects_reverse_dates() -> None:
    with pytest.raises(ValidationError, match="end_date"):
        DataPeriod(
            start_date=date(2025, 12, 31),
            end_date=date(2025, 1, 1),
        )


def test_negative_source_counts_are_rejected(
    analysis_period: DataPeriod,
    demo_source: SourceMetadata,
) -> None:
    with pytest.raises(ValidationError):
        TrafficRecord(
            airport_code="SFO",
            period=analysis_period,
            passengers=-1,
            source=demo_source,
        )


def test_operations_reject_performed_above_scheduled(
    analysis_period: DataPeriod,
    demo_source: SourceMetadata,
) -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        OperationalData(
            airport_code="SFO",
            period=analysis_period,
            scheduled_departures=100,
            performed_departures=101,
            source=demo_source,
        )


def test_route_origin_and_destination_must_differ(
    analysis_period: DataPeriod,
    demo_source: SourceMetadata,
) -> None:
    with pytest.raises(ValidationError, match="must differ"):
        RouteRecord(
            origin_airport_code="SFO",
            destination_airport_code="SFO",
            distance_miles=0,
            departures=1,
            passengers=1,
            period=analysis_period,
            source=demo_source,
        )


def test_tool_inputs_normalize_codes_and_enforce_limits() -> None:
    ranking = RankAirportsInput(
        region="New England",
        state_codes=["ma", "ri"],
        limit=5,
        excluded_airports=["bos"],
    )
    comparison = CompareAirportsInput(airport_codes=["lax", "sna"])
    long_haul = CalculateLongHaulShareInput(airport_code="anc")

    assert ranking.state_codes == ["MA", "RI"]
    assert ranking.excluded_airports == ["BOS"]
    assert comparison.airport_codes == ["LAX", "SNA"]
    assert long_haul.threshold_miles == 3000

    with pytest.raises(ValidationError):
        RankAirportsInput(limit=11)

    with pytest.raises(ValidationError, match="unique"):
        CompareAirportsInput(airport_codes=["LAX", "lax"])


def test_target_load_factor_is_bounded() -> None:
    assert EstimateUnmetCapacityInput(
        airport_code="SFO",
        target_load_factor=0.85,
    ).target_load_factor == 0.85

    with pytest.raises(ValidationError):
        EstimateUnmetCapacityInput(
            airport_code="SFO",
            target_load_factor=1.01,
        )


def test_scores_are_bounded_and_missing_metrics_are_supported() -> None:
    metrics = CalculatedMetrics(
        passenger_growth=0.08,
        congestion_score=77.5,
        investment_opportunity_score=72.0,
        missing_components=["departures_per_runway"],
    )

    assert metrics.congestion_score == 77.5
    assert metrics.missing_components == ["departures_per_runway"]

    with pytest.raises(ValidationError):
        CalculatedMetrics(investment_opportunity_score=101)


def test_unavailable_confidence_requires_zero_score() -> None:
    valid = ConfidenceInfo(
        level=ConfidenceLevel.UNAVAILABLE,
        score=0,
        reasons=["No fixture or public source was available."],
    )
    assert valid.score == 0

    with pytest.raises(ValidationError, match="score of 0"):
        ConfidenceInfo(
            level=ConfidenceLevel.UNAVAILABLE,
            score=0.2,
        )


def test_unavailable_data_response_is_typed() -> None:
    response = UnavailableDataResponse(
        tool_name="get_airport_profile",
        message="Airport is outside the supported MVP scope.",
        error_code="UNKNOWN_AIRPORT",
        airport_code="xyz",
    )

    assert response.airport_code == "XYZ"
    assert response.retryable is False
