"""Tests for strict domain and analytical-output contracts."""
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
    RankAirportsOutput,
    RankedAirport,
    RecommendationBand,
    RegionName,
    RouteRecord,
    SourceMetadata,
    TrafficRecord,
    UnavailableDataResponse,
)


@pytest.fixture
def current_period() -> DataPeriod:
    return DataPeriod(start_date=date(2025, 1, 1), end_date=date(2025, 12, 31))


@pytest.fixture
def previous_period() -> DataPeriod:
    return DataPeriod(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31))


def source(period: DataPeriod, mode: DataMode = DataMode.ILLUSTRATIVE_DEMO_DATA) -> SourceMetadata:
    return SourceMetadata(source_name="Test source", data_mode=mode, period=period)


def airport(period: DataPeriod) -> AirportRecord:
    return AirportRecord(
        airport_code="BOS",
        name="Boston Logan International Airport",
        city="Boston",
        state_code="MA",
        region="New England",
        source=source(period),
    )


def analysis(period: DataPeriod, previous: DataPeriod | None = None) -> AirportAnalysis:
    current_source = source(period)
    sources = [current_source]
    if previous is not None:
        sources.append(source(previous))
    airport_record = AirportRecord(
        airport_code="BOS",
        name="Boston Logan International Airport",
        city="Boston",
        state_code="MA",
        region="New England",
        source=current_source,
    )
    return AirportAnalysis(
        airport=airport_record,
        metrics=CalculatedMetrics(passenger_growth=0.08),
        confidence=ConfidenceInfo(level=ConfidenceLevel.HIGH, score=1),
        sources=sources,
    )


def test_airport_and_region_normalization(current_period: DataPeriod) -> None:
    record = AirportRecord(
        airport_code="sfo",
        name="San Francisco International Airport",
        city="San Francisco",
        state_code="ca",
        region="west",
        source=source(current_period),
    )
    assert record.airport_code == "SFO"
    assert record.state_code == "CA"
    assert record.region is RegionName.WEST
    ranking = RankAirportsInput(region="new england")
    assert ranking.state_codes == list(NEW_ENGLAND_STATE_CODES)

    with pytest.raises(ValidationError, match="unsupported region"):
        RankAirportsInput(region="Atlantis")
    with pytest.raises(ValidationError, match="valid US state"):
        RankAirportsInput(state_codes=["ZZ"])
    with pytest.raises(ValidationError, match="exactly"):
        RankAirportsInput(region="New England", state_codes=["MA"])


def test_period_and_previous_period_contracts(
    current_period: DataPeriod, previous_period: DataPeriod
) -> None:
    current_source = source(current_period)
    previous_source = source(previous_period)
    valid = TrafficRecord(
        airport_code="SFO",
        period=current_period,
        passengers=100,
        previous_period_passengers=90,
        previous_period=previous_period,
        previous_source=previous_source,
        available_seats=120,
        source=current_source,
    )
    assert valid.previous_period == previous_period

    with pytest.raises(ValidationError, match="supplied together"):
        TrafficRecord(
            airport_code="SFO",
            period=current_period,
            passengers=100,
            previous_period_passengers=90,
            source=current_source,
        )

    stale = DataPeriod(start_date=date(2020, 1, 1), end_date=date(2020, 12, 31))
    with pytest.raises(ValidationError, match="comparable"):
        TrafficRecord(
            airport_code="SFO",
            period=current_period,
            passengers=100,
            previous_period_passengers=80,
            previous_period=stale,
            previous_source=source(stale),
            source=current_source,
        )


def test_record_boundaries(current_period: DataPeriod) -> None:
    with pytest.raises(ValidationError, match="end_date"):
        DataPeriod(start_date=date(2025, 2, 1), end_date=date(2025, 1, 1))
    with pytest.raises(ValidationError, match="cannot exceed"):
        OperationalData(
            airport_code="SFO",
            period=current_period,
            scheduled_departures=10,
            performed_departures=11,
            source=source(current_period),
        )
    with pytest.raises(ValidationError, match="must differ"):
        RouteRecord(
            origin_airport_code="SFO",
            destination_airport_code="SFO",
            distance_miles=0,
            departures=1,
            passengers=1,
            period=current_period,
            source=source(current_period),
        )


def test_tool_input_contracts() -> None:
    comparison = CompareAirportsInput(
        airport_codes=["lax", "sna"],
        metrics=["congestion-score", "passenger_growth"],
    )
    assert comparison.airport_codes == ["LAX", "SNA"]
    assert [metric.value for metric in comparison.metrics or []] == [
        "congestion_score",
        "passenger_growth",
    ]
    with pytest.raises(ValidationError):
        RankAirportsInput(limit=11)
    with pytest.raises(ValidationError, match="unique"):
        CompareAirportsInput(airport_codes=["LAX", "lax"])
    with pytest.raises(ValidationError):
        CompareAirportsInput(airport_codes=["LAX", "SNA"], metrics=["unknown"])
    with pytest.raises(ValidationError, match="unique"):
        CompareAirportsInput(
            airport_codes=["LAX", "SNA"],
            metrics=["load_factor", "load_factor"],
        )
    with pytest.raises(ValidationError):
        EstimateUnmetCapacityInput(airport_code="SFO", target_load_factor=1.01)


def test_metric_and_recommendation_bounds() -> None:
    CalculatedMetrics(
        load_factor=0.9,
        average_departure_delay_minutes=10,
        average_taxi_out_minutes=12,
        investment_opportunity_score=80,
    )
    with pytest.raises(ValidationError):
        CalculatedMetrics(load_factor=1.01)
    with pytest.raises(ValidationError):
        CalculatedMetrics(investment_opportunity_score=101)
    with pytest.raises(ValueError):
        RecommendationBand("Guaranteed return")


def test_output_requires_complete_provenance(
    current_period: DataPeriod, previous_period: DataPeriod
) -> None:
    nested = analysis(current_period, previous_period)
    with pytest.raises(ValidationError, match="top-level sources"):
        AirportProfileOutput(
            input=GetAirportProfileInput(airport_code="BOS"),
            analysis=nested,
            data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
            period=current_period,
            sources=[source(current_period)],
        )
    output = AirportProfileOutput(
        input=GetAirportProfileInput(airport_code="BOS"),
        analysis=nested,
        data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
        period=current_period,
        sources=nested.sources,
    )
    assert len(output.sources) == 2

    stale_period = DataPeriod(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
    )
    stale_nested = analysis(current_period, stale_period)
    with pytest.raises(ValidationError, match="current or immediately previous period"):
        AirportProfileOutput(
            input=GetAirportProfileInput(airport_code="BOS"),
            analysis=stale_nested,
            data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
            period=current_period,
            sources=stale_nested.sources,
        )

    live = source(current_period, DataMode.LIVE_PUBLIC_DATA)
    with pytest.raises(ValidationError, match="data mode"):
        LongHaulShareOutput(
            input=CalculateLongHaulShareInput(airport_code="ANC"),
            long_haul_departures=1,
            all_departures=2,
            departure_share=0.5,
            long_haul_passengers=1,
            all_route_passengers=2,
            passenger_share=0.5,
            qualifying_routes=[],
            confidence=ConfidenceInfo(level=ConfidenceLevel.HIGH, score=1),
            data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
            period=current_period,
            sources=[live],
        )


def test_ranking_output_obeys_limit(current_period: DataPeriod) -> None:
    item = RankedAirport(
        rank=1,
        analysis=analysis(current_period),
        recommendation=RecommendationBand.MIXED,
    )
    second = item.model_copy(update={"rank": 2})
    with pytest.raises(ValidationError, match="input.limit"):
        RankAirportsOutput(
            input=RankAirportsInput(region="New England", limit=1),
            results=[item, second],
            data_mode=DataMode.ILLUSTRATIVE_DEMO_DATA,
            period=current_period,
            sources=item.analysis.sources,
        )


def test_typed_unavailable_response_and_confidence() -> None:
    response = UnavailableDataResponse(
        tool_name="get_airport_profile",
        message="Unsupported airport",
        error_code="UNKNOWN_AIRPORT",
        airport_code="xyz",
    )
    assert response.airport_code == "XYZ"
    with pytest.raises(ValidationError, match="score of 0"):
        ConfidenceInfo(level=ConfidenceLevel.UNAVAILABLE, score=0.1)
