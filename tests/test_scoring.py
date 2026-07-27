"""Tests for deterministic scoring and ranking behavior."""

from src.models import RecommendationBand
from src.scoring import (
    AirportScoringInput,
    deterministic_ranking_key,
    percentile_rank,
    recommendation_band,
    score_airports,
)


def _row(
    code: str,
    *,
    growth: float | None = 0.1,
    load: float | None = 0.8,
    delay: float | None = 10,
    taxi: float | None = 10,
    cancellation: float | None = 0.1,
    runway_pressure: float | None = 100,
    unmet_capacity: float | None = 1000,
    passengers: int = 10_000,
) -> AirportScoringInput:
    return AirportScoringInput(
        airport_code=code,
        passengers=passengers,
        passenger_growth=growth,
        load_factor=load,
        average_departure_delay_minutes=delay,
        average_taxi_out_minutes=taxi,
        cancellation_rate=cancellation,
        departures_per_runway=runway_pressure,
        estimated_unmet_capacity_proxy=unmet_capacity,
    )


def test_scores_are_bounded_and_repeatable() -> None:
    rows = [
        _row("AAA", growth=0.01, load=0.7, delay=5, taxi=6, cancellation=0.01,
             runway_pressure=50, unmet_capacity=0, passengers=100),
        _row("BBB", growth=0.2, load=0.95, delay=30, taxi=25, cancellation=0.2,
             runway_pressure=200, unmet_capacity=5000, passengers=10_000),
    ]
    first = score_airports(rows)
    second = score_airports(rows)
    assert first == second
    assert all(
        result.congestion_score is not None
        and 0 <= result.congestion_score <= 100
        and result.investment_opportunity_score is not None
        and 0 <= result.investment_opportunity_score <= 100
        for result in first.values()
    )


def test_missing_runway_is_renormalized_and_reduces_confidence() -> None:
    scores = score_airports([_row("AAA"), _row("BBB", runway_pressure=None)])
    assert scores["BBB"].congestion_score is not None
    assert "departures_per_runway" in scores["BBB"].missing_components
    assert scores["BBB"].confidence.score < scores["AAA"].confidence.score
    assert scores["BBB"].uncertainty_penalty == 4


def test_all_missing_congestion_inputs_are_penalized_once() -> None:
    scores = score_airports([
        _row("AAA"),
        _row(
            "BBB",
            delay=None,
            taxi=None,
            cancellation=None,
            runway_pressure=None,
        ),
    ])
    result = scores["BBB"]
    assert result.congestion_score is None
    assert result.missing_components == (
        "average_departure_delay_minutes",
        "average_taxi_out_minutes",
        "cancellation_rate",
        "departures_per_runway",
    )
    assert "congestion_score" not in result.missing_components
    assert result.uncertainty_penalty == 16
    assert result.confidence.score == 0.5


def test_recommendation_thresholds_are_exact() -> None:
    assert recommendation_band(75) is RecommendationBand.STRONG
    assert recommendation_band(60) is RecommendationBand.POTENTIAL
    assert recommendation_band(40) is RecommendationBand.MIXED
    assert recommendation_band(39.99) is RecommendationBand.WEAK


def test_ranking_uses_deterministic_tie_breakers() -> None:
    lower = _row("AAA", growth=0.1, passengers=100)
    higher = _row("BBB", growth=0.2, passengers=100)
    scores = score_airports([lower, higher])
    ordered = sorted([(lower, scores["AAA"]), (higher, scores["BBB"])], key=deterministic_ranking_key)
    assert ordered[0][0].airport_code == "BBB"


def test_percentile_ties_are_stable() -> None:
    assert percentile_rank(10, [10, 10]) == 50
