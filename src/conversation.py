"""Deterministic conversational context for approved airport tools."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agent import RouteSource, RoutingPolicy
from src.models import ComparisonMetric

CODES = frozenset(
    {"BOS", "BDL", "PVD", "MHT", "PWM", "BTV", "LAX", "SNA", "ANC", "SFO"}
)
ALIASES = {
    "boston": "BOS",
    "logan": "BOS",
    "bradley": "BDL",
    "hartford": "BDL",
    "providence": "PVD",
    "manchester": "MHT",
    "portland maine": "PWM",
    "burlington": "BTV",
    "los angeles": "LAX",
    "la airport": "LAX",
    "santa ana": "SNA",
    "john wayne": "SNA",
    "anchorage": "ANC",
    "san francisco": "SFO",
}
METRICS: dict[str, ComparisonMetric] = {
    "passenger growth": ComparisonMetric.PASSENGER_GROWTH,
    "load factor": ComparisonMetric.LOAD_FACTOR,
    "completion rate": ComparisonMetric.COMPLETION_RATE,
    "cancellation rate": ComparisonMetric.CANCELLATION_RATE,
    "cancellations": ComparisonMetric.CANCELLATION_RATE,
    "departures per runway": ComparisonMetric.DEPARTURES_PER_RUNWAY,
    "runway pressure": ComparisonMetric.DEPARTURES_PER_RUNWAY,
    "departure delay": ComparisonMetric.DEPARTURE_DELAY,
    "delay": ComparisonMetric.DEPARTURE_DELAY,
    "taxi out": ComparisonMetric.TAXI_OUT,
    "taxi time": ComparisonMetric.TAXI_OUT,
    "congestion score": ComparisonMetric.CONGESTION_SCORE,
    "congestion": ComparisonMetric.CONGESTION_SCORE,
    "opportunity score": ComparisonMetric.INVESTMENT_OPPORTUNITY_SCORE,
    "unmet capacity": ComparisonMetric.ESTIMATED_UNMET_CAPACITY_PROXY,
    "market scale": ComparisonMetric.MARKET_SCALE,
}

_INTEGER_TOKEN = r"([+-]?[0-9][0-9a-z_,.+-]*)"
_CANONICAL_INTEGER = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)")


class ConversationResolutionError(RuntimeError):
    pass


class ConversationState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    airport_codes: tuple[str, ...] = ()
    region: str | None = None
    metric: ComparisonMetric | None = None
    excluded_airports: tuple[str, ...] = ()
    target_load_factor: float = Field(default=0.82, gt=0, le=1)
    long_haul_threshold_miles: int = Field(default=3000, gt=0)
    ranking_limit: int = Field(default=5, ge=1, le=10)
    last_tool_name: str | None = None

    @classmethod
    def from_policy(cls, policy: RoutingPolicy) -> "ConversationState":
        return cls(
            target_load_factor=policy.target_load_factor,
            long_haul_threshold_miles=policy.long_haul_threshold_miles,
            ranking_limit=policy.ranking_limit,
        )

    @field_validator("airport_codes", "excluded_airports", mode="before")
    @classmethod
    def validate_codes(cls, values: Any) -> tuple[str, ...]:
        result: list[str] = []
        for value in values or ():
            code = str(value).strip().upper()
            if code not in CODES:
                raise ValueError(f"unsupported airport code: {code}")
            if code not in result:
                result.append(code)
        return tuple(result)


class Router(Protocol):
    def route(self, question: str) -> Any: ...


class Dispatcher(Protocol):
    def execute(
        self,
        *,
        question: str,
        tool_name: str,
        arguments: dict[str, Any],
        route_source: Any = ...,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    execution: Any
    state: ConversationState
    used_context: bool


def _text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.%+-]+", " ", value.casefold()).split())


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def extract_airport_codes(question: str) -> tuple[str, ...]:
    text = _text(question)
    found: list[tuple[int, str]] = []
    for match in re.finditer(r"(?<![a-z0-9])([a-z]{3})(?![a-z0-9])", text):
        code = match.group(1).upper()
        if code in CODES:
            found.append((match.start(), code))
    for alias, code in ALIASES.items():
        for match in re.finditer(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text
        ):
            found.append((match.start(), code))
    result: list[str] = []
    for _, code in sorted(found):
        if code not in result:
            result.append(code)
    return tuple(result)


def _metric(question: str) -> ComparisonMetric | None:
    text = _text(question)
    for phrase, value in sorted(METRICS.items(), key=lambda item: -len(item[0])):
        if _has(text, phrase):
            return value
    return None


def _percent(question: str) -> float | None:
    match = re.search(r"(?<![a-z0-9.])([+-]?\d+(?:\.\d+)?)\s*%", question.casefold())
    if not match:
        return None
    value = float(match.group(1)) / 100
    if not 0 < value <= 1:
        raise ConversationResolutionError("percentage must be greater than 0 and at most 100")
    return value


def _parse_canonical_integer(token: str, *, field_name: str) -> int:
    if _CANONICAL_INTEGER.fullmatch(token) is None:
        raise ConversationResolutionError(
            f"{field_name} must be a canonical whole number "
            "with optional sign and correctly grouped commas"
        )
    return int(token.replace(",", ""))


def _miles(question: str) -> int | None:
    match = re.search(
        rf"(?<![a-z0-9]){_INTEGER_TOKEN}\s*(?:statute\s+)?miles?\b",
        question.casefold(),
    )
    if not match:
        return None
    value = _parse_canonical_integer(match.group(1), field_name="mileage threshold")
    if value <= 0:
        raise ConversationResolutionError("mileage threshold must be greater than zero")
    return value


def _limit(question: str) -> int | None:
    match = re.search(
        rf"\b(?:top|first|limit(?:\s+to)?)\s+{_INTEGER_TOKEN}",
        question.casefold(),
    )
    if not match:
        return None
    value = _parse_canonical_integer(match.group(1), field_name="ranking limit")
    if not 1 <= value <= 10:
        raise ConversationResolutionError("ranking limit must be between 1 and 10")
    return value


class ConversationManager:
    def __init__(
        self,
        router: Router,
        dispatcher: Dispatcher,
        *,
        policy: RoutingPolicy | None = None,
    ) -> None:
        self.router = router
        self.dispatcher = dispatcher
        self.policy = policy or RoutingPolicy()

    def initial_state(self) -> ConversationState:
        return ConversationState.from_policy(self.policy)

    def handle(
        self, question: str, state: ConversationState | None = None
    ) -> ConversationTurn:
        current = state or self.initial_state()
        resolved = self._resolve(question, current)
        if resolved is None:
            execution = self.router.route(question.strip())
            used = False
        else:
            tool, arguments = resolved
            execution = self.dispatcher.execute(
                question=question.strip(),
                tool_name=tool,
                arguments=arguments,
                route_source=RouteSource.FALLBACK,
            )
            used = True
        return ConversationTurn(execution, self._update(current, execution), used)

    def _resolve(
        self, question: str, state: ConversationState
    ) -> tuple[str, dict[str, Any]] | None:
        text = _text(question)
        if not text:
            raise ConversationResolutionError("question must not be empty")
        codes, metric = extract_airport_codes(question), _metric(question)
        percent, miles, limit = _percent(question), _miles(question), _limit(question)
        exclude = any(
            _has(text, word) for word in ("exclude", "excluding", "without", "remove")
        )
        include = any(_has(text, word) for word in ("include", "add"))
        if exclude and include:
            raise ConversationResolutionError(
                "a follow-up cannot include and exclude airports at the same time"
            )
        if (exclude or include) and codes:
            if not state.region:
                raise ConversationResolutionError(
                    "airport exclusions require a previous regional ranking"
                )
            excluded = list(state.excluded_airports)
            for code in codes:
                if exclude and code not in excluded:
                    excluded.append(code)
                elif include and code in excluded:
                    excluded.remove(code)
            return "rank_airports", {
                "region": state.region,
                "limit": limit if limit is not None else state.ranking_limit,
                "excluded_airports": excluded,
            }
        if percent is not None and any(
            _has(text, word) for word in ("use", "target", "load factor")
        ):
            selected = codes or state.airport_codes
            if len(selected) != 1:
                raise ConversationResolutionError(
                    "target load-factor follow-up requires exactly one airport"
                )
            return "estimate_unmet_capacity", {
                "airport_code": selected[0],
                "target_load_factor": percent,
            }
        if miles is not None and any(
            _has(text, word) for word in ("use", "threshold", "long haul")
        ):
            selected = codes or state.airport_codes
            if len(selected) != 1:
                raise ConversationResolutionError(
                    "long-haul threshold follow-up requires exactly one airport"
                )
            return "calculate_long_haul_share", {
                "airport_code": selected[0],
                "threshold_miles": miles,
            }

        follow_up = any(
            phrase in text
            for phrase in (
                "what about",
                "how about",
                "compare them",
                "compare it",
                "same airports",
                "same metric",
            )
        )
        comparison_follow_up = follow_up and any(
            _has(text, word) for word in ("compare", "versus", "vs", "difference")
        )
        if comparison_follow_up:
            singular = _has(text, "it")
            plural = _has(text, "them") or _has(text, "same airports")
            if singular and len(state.airport_codes) != 1:
                raise ConversationResolutionError(
                    "'it' is ambiguous because multiple airports are stored"
                )
            if singular and len(codes) == 1:
                selected = [state.airport_codes[0], codes[0]]
            elif plural and state.airport_codes:
                selected = list(state.airport_codes)
                for code in codes:
                    if code not in selected:
                        selected.append(code)
            elif len(codes) >= 2:
                selected = list(codes)
            else:
                raise ConversationResolutionError(
                    "comparison follow-up does not identify enough airports"
                )
            args: dict[str, Any] = {"airport_codes": selected}
            if metric is not None:
                args["metrics"] = [metric.value]
            elif _has(text, "same metric") and state.metric is not None:
                args["metrics"] = [state.metric.value]
            return "compare_airports", args
        if metric and follow_up:
            if len(state.airport_codes) < 2:
                raise ConversationResolutionError(
                    "metric follow-up requires at least two stored airports"
                )
            return "compare_airports", {
                "airport_codes": list(state.airport_codes),
                "metrics": [metric.value],
            }
        if limit is not None and state.region and follow_up:
            return "rank_airports", {
                "region": state.region,
                "limit": limit,
                "excluded_airports": list(state.excluded_airports),
            }
        return None

    @staticmethod
    def _update(previous: ConversationState, execution: Any) -> ConversationState:
        tool, args = str(execution.tool_name), dict(execution.arguments)
        common: dict[str, Any] = {"last_tool_name": tool}
        if tool == "rank_airports":
            common.update(
                airport_codes=(),
                region=args.get("region") or previous.region,
                excluded_airports=tuple(args.get("excluded_airports") or ()),
                ranking_limit=args.get("limit", previous.ranking_limit),
                metric=None,
            )
        elif tool == "compare_airports":
            metrics = args.get("metrics") or []
            common.update(
                airport_codes=tuple(args.get("airport_codes") or ()),
                region=None,
                excluded_airports=(),
                metric=ComparisonMetric(metrics[0]) if metrics else None,
            )
        elif tool == "calculate_long_haul_share":
            common.update(
                airport_codes=(args["airport_code"],),
                region=None,
                excluded_airports=(),
                metric=None,
                long_haul_threshold_miles=args.get(
                    "threshold_miles", previous.long_haul_threshold_miles
                ),
            )
        elif tool == "estimate_unmet_capacity":
            common.update(
                airport_codes=(args["airport_code"],),
                region=None,
                excluded_airports=(),
                metric=None,
                target_load_factor=args.get(
                    "target_load_factor", previous.target_load_factor
                ),
            )
        elif tool == "get_airport_profile":
            common.update(
                airport_codes=(args["airport_code"],),
                region=None,
                excluded_airports=(),
                metric=None,
            )
        return previous.model_copy(update=common)
