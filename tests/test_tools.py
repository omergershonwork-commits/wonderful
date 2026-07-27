"""End-to-end tests for the five approved analytical tools."""

import pytest

from src.data.fixture_loader import FixtureDataBundle
from src.data.repository import FixtureAirportRepository
from src.exceptions import DataNotFoundError
from src.tools import (
    calculate_long_haul_share,
    compare_airports,
    estimate_unmet_capacity,
    get_airport_profile,
    rank_airports,
)


def test_new_england_ranking_is_deterministic() -> None:
    first = rank_airports(region="New England", limit=5)
    second = rank_airports(region="New England", limit=5)
    assert first.model_dump() == second.model_dump()
    assert len(first.results) == 5
    assert {r.analysis.airport.state_code for r in first.results} <= {
        "CT", "ME", "MA", "NH", "RI", "VT"
    }
    assert all(
        result.analysis.metrics.investment_opportunity_score is not None
        for result in first.results
    )


def test_valid_empty_ranking_returns_structured_empty_output() -> None:
    result = rank_airports(
        state_codes=["CA"],
        excluded_airports=["LAX", "SNA", "SFO"],
    )
    assert result.results == []
    assert result.sources


def test_lax_and_santa_ana_comparison_works_without_an_llm() -> None:
    result = compare_airports(["LAX", "SNA"])
    assert [a.airport.airport_code for a in result.airports] == ["LAX", "SNA"]
    assert all(a.metrics.congestion_score is not None for a in result.airports)


def test_comparison_metric_selector_is_honored() -> None:
    result = compare_airports(["LAX", "SNA"], metrics=["congestion_score"])
    for analysis in result.airports:
        assert analysis.metrics.congestion_score is not None
        assert analysis.metrics.passenger_growth is None
        assert analysis.metrics.load_factor is None
        assert analysis.metrics.average_departure_delay_minutes is None


def test_unsupported_codes_raise_typed_repository_error() -> None:
    with pytest.raises(DataNotFoundError, match="XYZ"):
        compare_airports(["LAX", "XYZ"])
    with pytest.raises(DataNotFoundError, match="XYZ"):
        get_airport_profile("XYZ")


def test_anchorage_long_haul_uses_complete_route_denominator() -> None:
    result = calculate_long_haul_share("ANC")
    assert result.all_departures == 13_400
    assert result.long_haul_departures == 3_800
    assert result.departure_share == pytest.approx(3_800 / 13_400)
    assert result.passenger_share == pytest.approx(554_000 / 1_784_000)
    assert {r.destination_airport_code for r in result.qualifying_routes} == {
        "DFW", "JFK", "NRT"
    }


def test_sfo_unmet_capacity_includes_previous_period_provenance() -> None:
    result = estimate_unmet_capacity("SFO")
    assert result.breakdown.raw_passenger_growth == pytest.approx(0.08)
    assert result.breakdown.projected_passengers == pytest.approx(58_320_000)
    assert result.breakdown.estimated_unmet_capacity_proxy == pytest.approx(
        58_320_000 / 0.82 - 63_000_000
    )
    assert any(
        source.period is not None and source.period.start_date.year == 2024
        for source in result.sources
    )
    assert "not observed lost demand" in result.assumptions[0]


def test_missing_congestion_inputs_do_not_crash_capacity_context() -> None:
    repository = FixtureAirportRepository.from_default_fixtures()
    bundle = repository.bundle
    operations = tuple(
        operation.model_copy(
            update={
                "scheduled_departures": 0,
                "performed_departures": 0,
                "reported_cancellations": None,
                "average_departure_delay_minutes": None,
                "average_taxi_out_minutes": None,
                "usable_runway_count": None,
            }
        )
        if operation.airport_code == "SFO"
        else operation
        for operation in bundle.operations
    )
    modified = FixtureDataBundle(
        airports=bundle.airports,
        traffic=bundle.traffic,
        routes=bundle.routes,
        operations=operations,
        period=bundle.period,
        data_mode=bundle.data_mode,
        disclaimer=bundle.disclaimer,
    )
    result = estimate_unmet_capacity(
        "SFO", repository=FixtureAirportRepository(modified)
    )
    assert "congestion score is unavailable" in result.assumptions[2]


def test_airport_profile_is_structured_and_grounded() -> None:
    result = get_airport_profile("BOS")
    assert result.analysis.airport.airport_code == "BOS"
    assert result.sources
    assert result.analysis.confidence.score > 0
