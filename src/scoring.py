"""Deterministic percentile-based airport scoring."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import inf

from src.models import ConfidenceInfo, ConfidenceLevel, RecommendationBand

CONGESTION_WEIGHTS = {
    "average_departure_delay_minutes": 0.40,
    "average_taxi_out_minutes": 0.25,
    "cancellation_rate": 0.20,
    "departures_per_runway": 0.15,
}
OPPORTUNITY_WEIGHTS = {
    "passenger_growth": 0.30,
    "load_factor": 0.25,
    "congestion_score": 0.20,
    "estimated_unmet_capacity_proxy": 0.15,
    "market_scale": 0.10,
}
UNCERTAINTY_PENALTY_PER_MISSING_COMPONENT = 4.0
MAX_UNCERTAINTY_PENALTY = 20.0


@dataclass(frozen=True, slots=True)
class AirportScoringInput:
    airport_code: str
    passengers: int
    passenger_growth: float | None
    load_factor: float | None
    average_departure_delay_minutes: float | None
    average_taxi_out_minutes: float | None
    cancellation_rate: float | None
    departures_per_runway: float | None
    estimated_unmet_capacity_proxy: float | None


@dataclass(frozen=True, slots=True)
class AirportScore:
    airport_code: str
    congestion_score: float | None
    investment_opportunity_score: float | None
    normalized_components: Mapping[str, float | None]
    missing_components: tuple[str, ...]
    uncertainty_penalty: float
    confidence: ConfidenceInfo
    recommendation: RecommendationBand


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    return max(minimum, min(maximum, value))


def percentile_rank(
    value: float | None,
    reference_values: Iterable[float | None],
) -> float | None:
    if value is None:
        return None
    values = sorted(float(item) for item in reference_values if item is not None)
    if not values:
        return None
    lower = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    return 100.0 * (lower + 0.5 * equal) / len(values)


def renormalized_weighted_score(
    components: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> tuple[float | None, tuple[str, ...]]:
    present = {
        name: value
        for name, value in components.items()
        if name in weights and value is not None
    }
    missing = tuple(name for name in weights if components.get(name) is None)
    included_weight = sum(weights[name] for name in present)
    if included_weight == 0:
        return None, missing
    score = sum(float(value) * weights[name] for name, value in present.items())
    return clamp(score / included_weight), missing


def confidence_from_missing(
    missing_components: Iterable[str], expected_component_count: int = 8
) -> ConfidenceInfo:
    unique = tuple(sorted(set(missing_components)))
    available = max(0, expected_component_count - len(unique))
    score = available / expected_component_count if expected_component_count else 0
    if score == 0:
        level = ConfidenceLevel.UNAVAILABLE
    elif score >= 0.875:
        level = ConfidenceLevel.HIGH
    elif score >= 0.625:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW
    reasons = (
        ["Confidence reduced because deterministic inputs are unavailable."]
        if unique
        else []
    )
    return ConfidenceInfo(
        level=level,
        score=score,
        missing_fields=list(unique),
        reasons=reasons,
    )


def recommendation_band(score: float | None) -> RecommendationBand:
    if score is None:
        return RecommendationBand.WEAK
    if score >= 75:
        return RecommendationBand.STRONG
    if score >= 60:
        return RecommendationBand.POTENTIAL
    if score >= 40:
        return RecommendationBand.MIXED
    return RecommendationBand.WEAK


def score_airports(inputs: Iterable[AirportScoringInput]) -> dict[str, AirportScore]:
    rows = tuple(inputs)
    if len({row.airport_code for row in rows}) != len(rows):
        raise ValueError("airport_code values must be unique")

    normalized: dict[str, dict[str, float | None]] = {
        row.airport_code: {} for row in rows
    }
    raw_fields = (
        "passenger_growth",
        "load_factor",
        "average_departure_delay_minutes",
        "average_taxi_out_minutes",
        "cancellation_rate",
        "departures_per_runway",
        "estimated_unmet_capacity_proxy",
    )
    for name in raw_fields:
        reference = [getattr(row, name) for row in rows]
        for row in rows:
            normalized[row.airport_code][name] = percentile_rank(
                getattr(row, name), reference
            )
    passenger_reference = [float(row.passengers) for row in rows]
    for row in rows:
        normalized[row.airport_code]["market_scale"] = percentile_rank(
            float(row.passengers), passenger_reference
        )

    congestion_scores: dict[str, float | None] = {}
    congestion_missing: dict[str, tuple[str, ...]] = {}
    for row in rows:
        score, missing = renormalized_weighted_score(
            normalized[row.airport_code], CONGESTION_WEIGHTS
        )
        congestion_scores[row.airport_code] = score
        congestion_missing[row.airport_code] = missing
        normalized[row.airport_code]["congestion_score"] = score

    results: dict[str, AirportScore] = {}
    for row in rows:
        opportunity = {
            name: normalized[row.airport_code].get(name)
            for name in OPPORTUNITY_WEIGHTS
        }
        base_score, opportunity_missing = renormalized_weighted_score(
            opportunity, OPPORTUNITY_WEIGHTS
        )
        # ``congestion_score`` is derived from the four raw congestion inputs.
        # Counting it again when all four are absent would double-penalize the
        # same missing evidence in both score and confidence.
        root_opportunity_missing = tuple(
            name for name in opportunity_missing if name != "congestion_score"
        )
        missing = tuple(
            sorted(
                set(
                    congestion_missing[row.airport_code]
                    + root_opportunity_missing
                )
            )
        )
        penalty = min(
            MAX_UNCERTAINTY_PENALTY,
            len(missing) * UNCERTAINTY_PENALTY_PER_MISSING_COMPONENT,
        )
        final = None if base_score is None else clamp(base_score - penalty)
        results[row.airport_code] = AirportScore(
            airport_code=row.airport_code,
            congestion_score=congestion_scores[row.airport_code],
            investment_opportunity_score=final,
            normalized_components=dict(normalized[row.airport_code]),
            missing_components=missing,
            uncertainty_penalty=penalty,
            confidence=confidence_from_missing(missing),
            recommendation=recommendation_band(final),
        )
    return results


def deterministic_ranking_key(
    item: tuple[AirportScoringInput, AirportScore],
) -> tuple[float, float, float, str]:
    raw, score = item
    return (
        -(score.investment_opportunity_score if score.investment_opportunity_score is not None else -inf),
        -(raw.passenger_growth if raw.passenger_growth is not None else -inf),
        -float(raw.passengers),
        raw.airport_code,
    )
