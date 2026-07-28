"""Deterministic parsing and semantic grounding of airport questions."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.agent_contracts import RoutingPolicy, SelectedToolCall, ToolArgumentsError
from src.models import (
    CalculateLongHaulShareInput,
    CompareAirportsInput,
    ComparisonMetric,
    EstimateUnmetCapacityInput,
    GetAirportProfileInput,
    RankAirportsInput,
    RegionName,
)

_AIRPORT_ALIASES: dict[str, tuple[str, ...]] = {
    "BOS": ("bos", "boston", "boston logan", "logan"),
    "BDL": ("bdl", "bradley", "hartford"),
    "PVD": ("pvd", "providence", "t f green", "tf green"),
    "MHT": ("mht", "manchester"),
    "PWM": ("pwm", "portland maine", "portland jetport"),
    "BTV": ("btv", "burlington"),
    "LAX": ("lax", "los angeles", "la airport", "la"),
    "SNA": ("sna", "santa ana", "john wayne"),
    "ANC": ("anc", "anchorage"),
    "SFO": ("sfo", "san francisco"),
}

_METRIC_PHRASES: dict[ComparisonMetric, tuple[str, ...]] = {
    ComparisonMetric.PASSENGER_GROWTH: ("passenger growth", "traffic growth"),
    ComparisonMetric.LOAD_FACTOR: ("load factor",),
    ComparisonMetric.COMPLETION_RATE: ("completion rate",),
    ComparisonMetric.CANCELLATION_RATE: ("cancellation rate", "cancellations"),
    ComparisonMetric.DEPARTURES_PER_RUNWAY: ("departures per runway", "runway pressure"),
    ComparisonMetric.DEPARTURE_DELAY: ("departure delay", "delays"),
    ComparisonMetric.TAXI_OUT: ("taxi out", "taxi-out", "taxi time"),
    ComparisonMetric.CONGESTION_SCORE: ("congestion",),
    ComparisonMetric.INVESTMENT_OPPORTUNITY_SCORE: (
        "opportunity score",
        "investment opportunity",
    ),
    ComparisonMetric.ESTIMATED_UNMET_CAPACITY_PROXY: (
        "unmet capacity",
        "unmet demand",
    ),
    ComparisonMetric.MARKET_SCALE: ("market scale", "passenger volume"),
}

_EXCLUSION_ACTIONS = frozenset({"exclude", "excluding", "without", "except", "remove"})
_INCLUSION_ACTIONS = frozenset({"include", "including", "with", "add", "keep"})
_ACTION_PATTERN = re.compile(
    r"(?<![a-z0-9])(exclude|excluding|without|except|remove|include|including|with|add|keep)(?![a-z0-9])"
)
_SIGNED_INTEGER = r"([+-]?\d[\d,]*)"
_SIGNED_NUMBER = r"([+-]?\d+(?:\.\d+)?)"


def _normalized_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9%+-]+", " ", value.casefold()).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = _normalized_text(phrase)
    return f" {normalized} " in f" {text} "


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _airport_mentions(question: str) -> tuple[tuple[int, str], ...]:
    text = _normalized_text(question)
    positions: list[tuple[int, str]] = []
    for code, aliases in _AIRPORT_ALIASES.items():
        hits = [
            text.find(_normalized_text(alias))
            for alias in aliases
            if _contains_phrase(text, alias)
        ]
        if hits:
            positions.append((min(hits), code))
    return tuple(sorted(positions))


def mentioned_airports(question: str) -> tuple[str, ...]:
    return tuple(code for _, code in _airport_mentions(question))


def _excluded_airports(question: str) -> tuple[str, ...]:
    """Bind each airport to the nearest preceding include/exclude action."""

    text = _normalized_text(question)
    actions = [(match.start(), match.group(1)) for match in _ACTION_PATTERN.finditer(text)]
    excluded: list[str] = []
    for position, code in _airport_mentions(question):
        preceding = [action for action in actions if action[0] < position]
        if not preceding:
            continue
        action = preceding[-1][1]
        if action in _EXCLUSION_ACTIONS and code not in excluded:
            excluded.append(code)
        elif action in _INCLUSION_ACTIONS and code in excluded:
            excluded.remove(code)
    return tuple(excluded)


def _rank_limit(question: str) -> int | None:
    for pattern in (
        rf"(?<![a-z0-9])top\s+{_SIGNED_INTEGER}(?![\d,])",
        rf"(?<![a-z0-9])limit(?:ed)?(?:\s+to)?\s+{_SIGNED_INTEGER}(?![\d,])",
        rf"(?<![a-z0-9])first\s+{_SIGNED_INTEGER}(?![\d,])",
        rf"(?<![a-z0-9]){_SIGNED_INTEGER}\s+(?:airports|candidates)(?![a-z0-9])",
    ):
        match = re.search(pattern, question.casefold())
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _threshold_miles(question: str) -> int | None:
    match = re.search(
        rf"(?<![a-z0-9]){_SIGNED_INTEGER}\s*(?:statute\s+)?miles?(?![a-z0-9])",
        question.casefold(),
    )
    return int(match.group(1).replace(",", "")) if match else None


def _target_load_factor(question: str) -> float | None:
    for pattern in (
        rf"(?:target\s+)?load\s+factor(?:\s+(?:of|at|to|=))?\s+{_SIGNED_NUMBER}(%?)",
        rf"{_SIGNED_NUMBER}(%)\s+(?:target\s+)?load\s+factor",
    ):
        match = re.search(pattern, question.casefold())
        if match:
            value = float(match.group(1))
            is_percent = match.group(2) == "%"
            if is_percent:
                return value / 100
            return value / 100 if abs(value) > 1 else value
    return None


def _requested_metrics(question: str) -> tuple[ComparisonMetric, ...]:
    text = _normalized_text(question)
    return tuple(
        metric
        for metric, phrases in _METRIC_PHRASES.items()
        if _contains_any(text, phrases)
    )


class QuestionConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    expected_tool: str
    airport_codes: tuple[str, ...] = ()
    region: RegionName | None = None
    requested_metrics: tuple[ComparisonMetric, ...] = ()
    excluded_airports: tuple[str, ...] = ()
    rank_limit: int | None = None
    threshold_miles: int | None = None
    target_load_factor: float | None = None

    def expected_arguments(self, policy: RoutingPolicy) -> dict[str, Any]:
        if self.expected_tool == "rank_airports":
            return {
                "region": (self.region or RegionName.NEW_ENGLAND).value,
                "limit": (
                    self.rank_limit
                    if self.rank_limit is not None
                    else policy.ranking_limit
                ),
                "excluded_airports": list(self.excluded_airports),
            }
        if self.expected_tool == "compare_airports":
            result: dict[str, Any] = {"airport_codes": list(self.airport_codes)}
            if self.requested_metrics:
                result["metrics"] = [item.value for item in self.requested_metrics]
            return result
        if self.expected_tool == "calculate_long_haul_share":
            return {
                "airport_code": self.airport_codes[0],
                "threshold_miles": (
                    self.threshold_miles
                    if self.threshold_miles is not None
                    else policy.long_haul_threshold_miles
                ),
            }
        if self.expected_tool == "estimate_unmet_capacity":
            return {
                "airport_code": self.airport_codes[0],
                "target_load_factor": (
                    self.target_load_factor
                    if self.target_load_factor is not None
                    else policy.target_load_factor
                ),
            }
        if self.expected_tool == "get_airport_profile":
            return {"airport_code": self.airport_codes[0]}
        raise ToolArgumentsError("Unsupported deterministic tool intent")


def parse_question_constraints(question: str) -> QuestionConstraints:
    text = _normalized_text(question)
    if not text:
        raise ToolArgumentsError("question must not be empty")
    airports = mentioned_airports(question)

    if _contains_phrase(text, "new england") and _contains_any(
        text,
        (
            "rank",
            "candidate",
            "candidates",
            "terminal expansion",
            "expansion",
            "expand",
            "investment",
        ),
    ):
        return QuestionConstraints(
            expected_tool="rank_airports",
            region=RegionName.NEW_ENGLAND,
            excluded_airports=_excluded_airports(question),
            rank_limit=_rank_limit(question),
        )
    if len(airports) >= 2 and _contains_any(
        text, ("compare", "versus", "vs", "difference", "congestion")
    ):
        return QuestionConstraints(
            expected_tool="compare_airports",
            airport_codes=airports,
            requested_metrics=_requested_metrics(question),
        )
    if len(airports) == 1 and _contains_any(
        text, ("long haul", "longhaul", "long distance")
    ):
        return QuestionConstraints(
            expected_tool="calculate_long_haul_share",
            airport_codes=airports,
            threshold_miles=_threshold_miles(question),
        )
    if len(airports) == 1 and _contains_any(
        text, ("unmet", "flight demand", "seat demand", "capacity", "lost demand")
    ):
        return QuestionConstraints(
            expected_tool="estimate_unmet_capacity",
            airport_codes=airports,
            target_load_factor=_target_load_factor(question),
        )
    if len(airports) == 1 and _contains_any(
        text, ("profile", "overview", "details", "information", "tell me about")
    ):
        return QuestionConstraints(
            expected_tool="get_airport_profile", airport_codes=airports
        )
    raise ToolArgumentsError(
        "The question does not establish a supported deterministic tool intent"
    )


def apply_numeric_policy_defaults(
    selected: SelectedToolCall,
    constraints: QuestionConstraints,
    policy: RoutingPolicy,
) -> dict[str, Any]:
    arguments = dict(selected.arguments)
    expected = constraints.expected_arguments(policy)
    if selected.name == "rank_airports":
        arguments.setdefault("limit", expected["limit"])
    elif selected.name == "calculate_long_haul_share":
        arguments.setdefault("threshold_miles", expected["threshold_miles"])
    elif selected.name == "estimate_unmet_capacity":
        arguments.setdefault("target_load_factor", expected["target_load_factor"])
    return arguments


def validate_question_semantics(
    constraints: QuestionConstraints,
    tool_name: str,
    request: BaseModel,
    policy: RoutingPolicy,
) -> None:
    if tool_name != constraints.expected_tool:
        raise ToolArgumentsError(
            f"Selected tool {tool_name} contradicts requested intent {constraints.expected_tool}"
        )
    expected = constraints.expected_arguments(policy)
    if isinstance(request, RankAirportsInput):
        if request.region is not constraints.region or request.limit != expected["limit"]:
            raise ToolArgumentsError("Ranking region or limit contradicts the question")
        if set(request.excluded_airports) != set(constraints.excluded_airports):
            raise ToolArgumentsError(
                "Ranking exclusions must exactly match the named airports"
            )
        return
    if isinstance(request, CompareAirportsInput):
        if tuple(request.airport_codes) != constraints.airport_codes:
            raise ToolArgumentsError(
                "Comparison airports must match the question in order"
            )
        if tuple(request.metrics or ()) != constraints.requested_metrics:
            raise ToolArgumentsError(
                "Comparison metric selector must exactly match explicitly requested metrics"
            )
        return
    if isinstance(request, CalculateLongHaulShareInput):
        if (
            request.airport_code != constraints.airport_codes[0]
            or request.threshold_miles != expected["threshold_miles"]
        ):
            raise ToolArgumentsError(
                "Long-haul airport or threshold contradicts the question"
            )
        return
    if isinstance(request, EstimateUnmetCapacityInput):
        if (
            request.airport_code != constraints.airport_codes[0]
            or abs(
                request.target_load_factor - expected["target_load_factor"]
            )
            > 1e-12
        ):
            raise ToolArgumentsError(
                "Capacity airport or target load factor contradicts the question"
            )
        return
    if isinstance(request, GetAirportProfileInput):
        if request.airport_code != constraints.airport_codes[0]:
            raise ToolArgumentsError("Profile airport must match the question")
        return
    raise ToolArgumentsError("Unsupported validated request type")
