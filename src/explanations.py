"""Deterministic user-visible explanations for structured airport tool results.

The MVP deliberately does not display arbitrary LLM-generated prose. All visible
claims are rendered from typed deterministic tool output, including provenance.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.agent import LMStudioChatClient, ToolExecutionResult


class ExplanationError(RuntimeError):
    """Raised when a structured tool result cannot be rendered safely."""


class ExplanationSource(StrEnum):
    """Stable generation-source values retained for API compatibility."""

    LLM = "llm"
    TEMPLATE = "template"


class Explanation(BaseModel):
    """Application-facing deterministic explanation and provenance summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    source: ExplanationSource
    source_mode: str = Field(min_length=1)
    assumptions: tuple[str, ...] = ()


def _output_payload(output: Any) -> dict[str, Any]:
    if isinstance(output, BaseModel):
        payload = output.model_dump(mode="json")
    elif isinstance(output, dict):
        payload = output
    else:
        raise ExplanationError("Tool output must be a Pydantic model or JSON object")
    if not isinstance(payload, dict):
        raise ExplanationError("Serialized tool output must be an object")
    return payload


def _collect_assumptions(value: Any) -> list[str]:
    collected: list[str] = []
    if isinstance(value, dict):
        raw = value.get("assumptions")
        if isinstance(raw, list):
            collected.extend(str(item) for item in raw)
        for item in value.values():
            collected.extend(_collect_assumptions(item))
    elif isinstance(value, list):
        for item in value:
            collected.extend(_collect_assumptions(item))
    return collected


def _collect_sources(value: Any) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    if isinstance(value, dict):
        raw = value.get("sources")
        if isinstance(raw, list):
            collected.extend(item for item in raw if isinstance(item, dict))
        for key, item in value.items():
            if key != "sources":
                collected.extend(_collect_sources(item))
    elif isinstance(value, list):
        for item in value:
            collected.extend(_collect_sources(item))
    return collected


def _source_key(source: dict[str, Any]) -> tuple[Any, ...]:
    period = source.get("period") if isinstance(source.get("period"), dict) else {}
    return (
        source.get("source_name"),
        source.get("data_mode"),
        source.get("retrieved_at"),
        period.get("start_date"),
        period.get("end_date"),
        source.get("source_url"),
    )


def _dedupe_sources(sources: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for source in sources:
        key = _source_key(source)
        if key not in seen:
            seen.add(key)
            result.append(source)
    return tuple(result)


def _display(value: Any) -> str:
    return "unavailable" if value is None else str(value)


def _format_period(period: Any) -> str:
    if not isinstance(period, dict):
        return "unavailable"
    start = period.get("start_date")
    end = period.get("end_date")
    label = period.get("label")
    if start and end:
        rendered = f"{start} to {end}"
        return f"{rendered} ({label})" if label else rendered
    return str(label or "unavailable")


def _retrieval_date(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "unavailable"
    return value.split("T", 1)[0]


def _template_body(tool_name: str, payload: dict[str, Any]) -> str:
    """Render one deterministic, tool-specific explanation without inference."""

    lines: list[str] = []
    if tool_name == "rank_airports":
        lines.append("Deterministic airport ranking:")
        results = payload.get("results", [])
        if not results:
            lines.append("- No airports matched the validated filters.")
        for item in results:
            analysis = item.get("analysis", {})
            airport = analysis.get("airport", {})
            metrics = analysis.get("metrics", {})
            confidence = analysis.get("confidence", {})
            lines.append(
                "- Rank {rank}: {code}; opportunity score {score}; "
                "recommendation {recommendation}; confidence {confidence}.".format(
                    rank=_display(item.get("rank")),
                    code=_display(airport.get("airport_code")),
                    score=_display(metrics.get("investment_opportunity_score")),
                    recommendation=_display(item.get("recommendation")),
                    confidence=_display(confidence.get("level")),
                )
            )
    elif tool_name == "compare_airports":
        lines.append("Deterministic airport comparison:")
        for analysis in payload.get("airports", []):
            airport = analysis.get("airport", {})
            metrics = analysis.get("metrics", {})
            lines.append(
                "- {code}: congestion score {congestion}; opportunity score {opportunity}; "
                "passenger growth {growth}; load factor {load}.".format(
                    code=_display(airport.get("airport_code")),
                    congestion=_display(metrics.get("congestion_score")),
                    opportunity=_display(metrics.get("investment_opportunity_score")),
                    growth=_display(metrics.get("passenger_growth")),
                    load=_display(metrics.get("load_factor")),
                )
            )
    elif tool_name == "calculate_long_haul_share":
        lines.extend(
            [
                "Deterministic long-haul result:",
                f"- Long-haul departures: {_display(payload.get('long_haul_departures'))}.",
                f"- All departures: {_display(payload.get('all_departures'))}.",
                f"- Departure share: {_display(payload.get('departure_share'))}.",
                f"- Long-haul passengers: {_display(payload.get('long_haul_passengers'))}.",
                f"- All route passengers: {_display(payload.get('all_route_passengers'))}.",
                f"- Passenger share: {_display(payload.get('passenger_share'))}.",
            ]
        )
    elif tool_name == "estimate_unmet_capacity":
        breakdown = payload.get("breakdown", {})
        lines.extend(
            [
                "Deterministic unmet-capacity proxy:",
                f"- Current passengers: {_display(breakdown.get('current_passengers'))}.",
                f"- Current available seats: {_display(breakdown.get('current_available_seats'))}.",
                f"- Raw passenger growth: {_display(breakdown.get('raw_passenger_growth'))}.",
                f"- Clamped passenger growth: {_display(breakdown.get('clamped_passenger_growth'))}.",
                f"- Projected passengers: {_display(breakdown.get('projected_passengers'))}.",
                f"- Target load factor: {_display(breakdown.get('target_load_factor'))}.",
                f"- Required seats: {_display(breakdown.get('required_seats'))}.",
                "- Estimated unmet-capacity proxy: "
                f"{_display(breakdown.get('estimated_unmet_capacity_proxy'))}.",
            ]
        )
    elif tool_name == "get_airport_profile":
        analysis = payload.get("analysis", {})
        airport = analysis.get("airport", {})
        metrics = analysis.get("metrics", {})
        confidence = analysis.get("confidence", {})
        lines.extend(
            [
                f"Deterministic profile for {_display(airport.get('airport_code'))}:",
                f"- Airport: {_display(airport.get('name'))}.",
                f"- Congestion score: {_display(metrics.get('congestion_score'))}.",
                "- Investment opportunity score: "
                f"{_display(metrics.get('investment_opportunity_score'))}.",
                f"- Confidence: {_display(confidence.get('level'))}.",
            ]
        )
    else:
        lines.extend(
            [
                "Deterministic tool result:",
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            ]
        )
    return "\n".join(lines)


def _provenance_footer(payload: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    source_mode = str(payload.get("data_mode") or "UNKNOWN SOURCE MODE")
    assumptions = tuple(dict.fromkeys(_collect_assumptions(payload)))
    sources = _dedupe_sources(_collect_sources(payload))
    output_period = payload.get("period")

    lines = [
        "Provenance:",
        f"- Source mode: {source_mode}.",
        f"- Analysis period: {_format_period(output_period)}.",
        "- Sources:",
    ]
    if sources:
        for source in sources:
            lines.append(
                "  - {name}; retrieved/fixture date {retrieved}; source period {period}.".format(
                    name=_display(source.get("source_name")),
                    retrieved=_retrieval_date(source.get("retrieved_at")),
                    period=_format_period(source.get("period")),
                )
            )
    else:
        lines.append("  - No source metadata was present in the structured result.")

    lines.append("Assumptions and limitations:")
    if assumptions:
        lines.extend(f"- {item}" for item in assumptions)
    else:
        lines.append("- No assumptions were stated in the structured result.")
    return "\n".join(lines), source_mode, assumptions


class ExplanationGenerator:
    """Render deterministic templates; never display arbitrary model prose."""

    def __init__(self, client: LMStudioChatClient | None) -> None:
        # Retained for constructor compatibility. The MVP does not call the model
        # for user-visible explanations because semantic claim validation is not
        # reliable enough for arbitrary prose.
        self._client = client

    def generate(self, execution: ToolExecutionResult) -> Explanation:
        payload = _output_payload(execution.output)
        body = _template_body(execution.tool_name, payload)
        footer, source_mode, assumptions = _provenance_footer(payload)
        return Explanation(
            text=body + "\n\n" + footer,
            source=ExplanationSource.TEMPLATE,
            source_mode=source_mode,
            assumptions=assumptions,
        )
