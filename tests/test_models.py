"""Tests for normalized domain models and approved tool boundaries."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.models import (
    AirportAnalysis,
    AirportProfileOutput,
    AirportRecord,
    CalculateLongHaulShareInput,
    CalculatedMetrics,
    CompareAirportsInput,
    ConfidenceInfo,
    ConfidenceLevel,
    DataMode,
    DataPeriod,
    EstimateUnmetCapacityInput,
    GetAirportProfileInput,
    LongHaulShareOutput,
    NEW_ENGLAND_STATE_CODES,
    OperationalData,
    RankAirportsInput,
    RecommendationBand,
    RegionName,
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
def previous_period() -> DataPeriod:
    return DataPeriod(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        label="Calendar year 2024",
    )


@pytest.fixture
def demo_source(analysis_period: DataPeriod) -> SourceMetadata:
    return SourceMetadata(
        source_name="Illustrative Phase 3 fixture",
        data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
        period=analysis_period,
        notes=["Values exist only to validate the normalized schema."],
    )


@pytest.fixture
def previous_demo_source(previous_period: DataPeriod) -> SourceMetadata:
    return SourceMetadata(
        source_name="Illustrative Phase 3 fixture",
        data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
        period=previous_period,
    )


def test_sfo_fixture_validates(
    analysis_period: DataPeriod,
    previous_period: DataPeriod,
    demo_source: SourceMetadata,
    previous_demo_source: SourceMetadata,
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
        previous_period=previous_period,
        previous_source=previous_demo_source,
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
    assert airport.region is RegionName.WEST
    assert traffic.previous_period == previous_period
    assert operations.performed_departures == 182_000
    assert route.destination_airport_code == "JFK"


def test_invalid_airport_code_is_rejected(demo_source: SourceMetadata) -> None:
    with pytest.raises(ValidationError, match="exactly three letters"):
        AirportRecord(
            airport_code="SF",
            name="Invalid Airport",
            city="San Francisco",
            state_code="CA",
            source=demo_source,
        )


def test_region_and_state_semantics_are_enforced(demo_source: SourceMetadata) -> None:
    ranking = RankAirportsInput(region="new england")
    assert ranking.region is RegionName.NEW_ENGLAND
    assert ranking.state_codes == list(NEW_ENGLAND_STATE_CODES)

    with pytest.raises(ValidationError, match="unsupported region"):
        RankAirportsInput(region="Atlantis")

    with pytest.raises(ValidationError, match="valid US state"):
        RankAirportsInput(state_codes=["ZZ"])

    with pytest.raises(ValidationError, match="exactly"):
        RankAirportsInput(region="New England", state_codes=["MA", "RI"])

    with pytest.raises(ValidationError, match="New England airports"):
        AirportRecord(
            airport_code="SFO",
            name="San Francisco International Airport",
            city="San Francisco",
            state_code="CA",
            region="New England",
            source=demo_source,
        )


def test_data_period_rejects_reverse_dates() -> None:
    with pytest.raises(ValidationError, match="end_date"):
        DataPeriod(
            start_date=date(2025, 12, 31),
            end_date=date(2025, 1, 1),
        )


def test_previous_passenger_metadata_is_required_and_comparable(
    analysis_period: DataPeriod,
    previous_period: DataPeriod,
    demo_source: SourceMetadata,
    previous_demo_source: SourceMetadata,
) -> None:
    with pytest.raises(ValidationError, match="supplied together"):
        TrafficRecord(
            airport_code="SFO",
            period=analysis_period,
            passengers=52_000_000,
            previous_period_passengers=49_000_000,
            source=demo_source,
        )

    partial_previous = DataPeriod(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )
    partial_source = SourceMetadata(
        source_name="Partial fixture",
        data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
        period=partial_previous,
    )
    with pytest.raises(ValidationError, match="comparable"):
        TrafficRecord(
            airport_code="SFO",
            period=analysis_period,
            passengers=52_000_000,
            previous_period_passengers=25_000_000,
            previous_period=partial_previous,
            previous_source=partial_source,
            source=demo_source,
        )

    valid = TrafficRecord(
        airport_code="SFO",
        period=analysis_period,
        passengers=52_000_000,
        previous_period_passengers=49_000_000,
        previous_period=previous_period,
        previous_source=previous_demo_source,
        source=demo_source,
    )
    assert valid.previous_period_passengers == 49_000_000


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
        limit=5,
        excluded_airports=["bos"],
    )
    comparison = CompareAirportsInput(
        airport_codes=["lax", "sna"],
        metrics=["congestion-score", "passenger_growth"],
    )
    long_haul = CalculateLongHaulShareInput(airport_code="anc")

    assert ranking.state_codes == list(NEW_ENGLAND_STATE_CODES)
    assert ranking.excluded_airports == ["BOS"]
    assert comparison.airport_codes == ["LAX", "SNA"]
    assert [metric.value for metric in comparison.metrics or []] == [
        "congestion_score",
        "passenger_growth",
    ]
    assert long_haul.threshold_miles == 3000

    with pytest.raises(ValidationError):
        RankAirportsInput(limit=11)

    with pytest.raises(ValidationError, match="unique"):
        CompareAirportsInput(airport_codes=["LAX", "lax"])

    with pytest.raises(ValidationError):
        CompareAirportsInput(airport_codes=["LAX", "SNA"], metrics=["made_up"])

    with pytest.raises(ValidationError, match="unique"):
        CompareAirportsInput(
            airport_codes=["LAX", "SNA"],
            metrics=["load_factor", "load_factor"],
        )


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


def test_scores_and_load_factor_are_bounded() -> None:
    metrics = CalculatedMetrics(
        passenger_growth=0.08,
        load_factor=0.87,
        congestion_score=77.5,
        investment_opportunity_score=72.0,
        missing_components=["departures_per_runway"],
    )

    assert metrics.load_factor == 0.87
    assert metrics.missing_components == ["departures_per_runway"]

    with pytest.raises(ValidationError):
        CalculatedMetrics(investment_opportunity_score=101)
    with pytest.raises(ValidationError):
        CalculatedMetrics(load_factor=1.01)


def test_recommendation_band_rejects_free_form_text() -> None:
    assert RecommendationBand.POTENTIAL.value == "Potential candidate"
    with pytest.raises(ValueError):
        RecommendationBand("Guaranteed profit")


def test_analytical_outputs_require_consistent_complete_provenance(
    analysis_period: DataPeriod,
    demo_source: SourceMetadata,
) -> None:
    confidence = ConfidenceInfo(level=ConfidenceLevel.HIGH, score=0.9)
    common = dict(
        input=CalculateLongHaulShareInput(airport_code="ANC"),
        long_haul_departures=10,
        all_departures=20,
        departure_share=0.5,
        long_haul_passengers=100,
        all_route_passengers=200,
        passenger_share=0.5,
        qualifying_routes=[],
        confidence=confidence,
        data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
        period=analysis_period,
    )

    with pytest.raises(ValidationError):
        LongHaulShareOutput(**common)

    live_source = SourceMetadata(
        source_name="Live source",
        data_mode=DataMode.LIVE_PUBLIC_DATA,
        period=analysis_period,
    )
    with pytest.raises(ValidationError, match="data mode"):
        LongHaulShareOutput(**common, sources=[live_source])

    wrong_period = DataPeriod(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )
    wrong_period_source = SourceMetadata(
        source_name="Wrong-period source",
        data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
        period=wrong_period,
    )
    with pytest.raises(ValidationError, match="period"):
        LongHaulShareOutput(**common, sources=[wrong_period_source])

    output = LongHaulShareOutput(**common, sources=[demo_source])
    assert output.sources == [demo_source]


def test_nested_source_cannot_contradict_output_mode(
    analysis_period: DataPeriod,
    demo_source: SourceMetadata,
) -> None:
    airport = AirportRecord(
        airport_code="SFO",
        name="San Francisco International Airport",
        city="San Francisco",
        state_code="CA",
        region="West",
        source=demo_source,
    )
    analysis = AirportAnalysis(
        airport=airport,
        metrics=CalculatedMetrics(),
        confidence=ConfidenceInfo(level=ConfidenceLevel.HIGH, score=0.9),
        sources=[demo_source],
    )
    live_source = SourceMetadata(
        source_name="Live source",
        data_mode=DataMode.LIVE_PUBLIC_DATA,
        period=analysis_period,
    )
    with pytest.raises(ValidationError, match="nested source data mode"):
        AirportProfileOutput(
            input=GetAirportProfileInput(airport_code="SFO"),
            analysis=analysis,
            data_mode=DataMode.LIVE_PUBLIC_DATA,
            period=analysis_period,
            sources=[live_source],
        )


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
