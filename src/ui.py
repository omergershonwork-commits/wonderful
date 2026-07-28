"""Pure presentation adapters for the Streamlit interface."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

SUGGESTED_PROMPTS = (
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LAX and SNA airport congestion levels.",
    "What percentage of long-haul flights leave ANC?",
    "What is the unmet flight demand at SFO and why?",
)


class UiCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    help_text: str | None = None


class UiResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    heading: str
    cards: tuple[UiCard, ...] = ()
    table_rows: tuple[dict[str, Any], ...] = ()
    metric_rows: tuple[dict[str, Any], ...] = ()
    source_mode: str = "UNKNOWN SOURCE MODE"
    confidence: str = "UNAVAILABLE"
    analysis_period: str = "Unavailable"
    sources: tuple[dict[str, str], ...] = ()
    assumptions: tuple[str, ...] = ()


class ModelStatusView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str
    detail: str
    severity: str


def _payload(output: Any) -> dict[str, Any]:
    if isinstance(output, BaseModel):
        output = output.model_dump(mode="json")
    if not isinstance(output, dict):
        raise TypeError("tool output must serialize to an object")
    return output


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if percent:
            return f"{value * 100:.2f}%"
        return f"{value:,.2f}" if not value.is_integer() else f"{int(value):,}"
    return str(value)


def _period(value: Any) -> str:
    if not isinstance(value, dict):
        return "Unavailable"
    return str(value.get("label") or f"{value.get('start_date')} to {value.get('end_date')}")


def _sources(payload: dict[str, Any]) -> tuple[dict[str, str], ...]:
    rows, seen = [], set()
    for source in payload.get("sources") or []:
        if not isinstance(source, dict):
            continue
        row = {
            "Source": str(source.get("source_name") or "Unnamed source"),
            "Mode": str(source.get("data_mode") or payload.get("data_mode") or "UNKNOWN"),
            "Retrieved / fixture date": str(source.get("retrieved_at") or "Unavailable"),
            "Source period": _period(source.get("period")),
        }
        key = tuple(row.values())
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return tuple(rows)


def _assumptions(value: Any) -> tuple[str, ...]:
    result: list[str] = []
    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if isinstance(item.get("assumptions"), list):
                result.extend(str(v) for v in item["assumptions"])
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return tuple(dict.fromkeys(result))


def _confidence(payload: dict[str, Any]) -> str:
    for candidate in (payload.get("confidence"), (payload.get("analysis") or {}).get("confidence")):
        if isinstance(candidate, dict) and candidate.get("level"):
            return str(candidate["level"])
    levels = []
    for key in ("results", "airports"):
        for item in payload.get(key) or []:
            analysis = item.get("analysis", item) if isinstance(item, dict) else {}
            confidence = analysis.get("confidence") if isinstance(analysis, dict) else None
            if isinstance(confidence, dict) and confidence.get("level"):
                levels.append(str(confidence["level"]))
    return ", ".join(dict.fromkeys(levels)) or "UNAVAILABLE"


def _metrics(values: dict[str, Any]) -> tuple[dict[str, str], ...]:
    percentages = {"passenger_growth", "load_factor", "completion_rate", "cancellation_rate", "target_load_factor"}
    return tuple(
        {"Metric": key.replace("_", " ").title(), "Value": _fmt(value, key in percentages)}
        for key, value in values.items() if key != "missing_components"
    )


def build_ui_result(execution: Any) -> UiResult:
    payload, tool = _payload(execution.output), str(execution.tool_name)
    cards: list[UiCard] = []
    rows: list[dict[str, Any]] = []
    metric_rows: tuple[dict[str, Any], ...] = ()
    heading = tool.replace("_", " ").title()

    if tool == "rank_airports":
        heading = "Ranked airport expansion candidates"
        for item in payload.get("results") or []:
            analysis, airport = item.get("analysis", {}), item.get("analysis", {}).get("airport", {})
            metrics, confidence = analysis.get("metrics", {}), analysis.get("confidence", {})
            rows.append({"Rank": item.get("rank"), "Airport": airport.get("airport_code"), "Name": airport.get("name"), "State": airport.get("state_code"), "Opportunity score": metrics.get("investment_opportunity_score"), "Congestion score": metrics.get("congestion_score"), "Confidence": confidence.get("level"), "Recommendation": item.get("recommendation")})
        cards.append(UiCard(label="Candidates returned", value=str(len(rows))))
    elif tool == "compare_airports":
        heading = "Airport comparison"
        percentages = {"passenger_growth", "load_factor", "completion_rate", "cancellation_rate"}
        for analysis in payload.get("airports") or []:
            airport, metrics = analysis.get("airport", {}), analysis.get("metrics", {})
            row: dict[str, Any] = {"Airport": airport.get("airport_code"), "Name": airport.get("name")}
            for key, value in metrics.items():
                if key == "missing_components":
                    row["Missing Components"] = ", ".join(value or ()) or "None"
                else:
                    row[key.replace("_", " ").title()] = _fmt(value, key in percentages)
            rows.append(row)
        cards.append(UiCard(label="Airports compared", value=str(len(rows))))
    elif tool == "calculate_long_haul_share":
        heading = "Long-haul share"
        cards = [UiCard(label="Flight-weighted share", value=_fmt(payload.get("departure_share"), True)), UiCard(label="Passenger-weighted share", value=_fmt(payload.get("passenger_share"), True)), UiCard(label="Long-haul departures", value=_fmt(payload.get("long_haul_departures"))), UiCard(label="All departures", value=_fmt(payload.get("all_departures")))]
        rows = list(payload.get("qualifying_routes") or [])
    elif tool == "estimate_unmet_capacity":
        heading, breakdown = "Estimated unmet-capacity proxy", payload.get("breakdown") or {}
        cards = [UiCard(label="Estimated unmet capacity", value=_fmt(breakdown.get("estimated_unmet_capacity_proxy")), help_text="Deterministic screening proxy, not observed lost demand."), UiCard(label="Projected passengers", value=_fmt(breakdown.get("projected_passengers"))), UiCard(label="Target load factor", value=_fmt(breakdown.get("target_load_factor"), True))]
        metric_rows = _metrics(breakdown)
    elif tool == "get_airport_profile":
        analysis, airport = payload.get("analysis") or {}, (payload.get("analysis") or {}).get("airport") or {}
        metrics = analysis.get("metrics") or {}
        heading = f"Airport profile: {airport.get('airport_code', 'Unknown')}"
        cards = [UiCard(label="Airport", value=str(airport.get("name") or "Unavailable")), UiCard(label="Opportunity score", value=_fmt(metrics.get("investment_opportunity_score"))), UiCard(label="Congestion score", value=_fmt(metrics.get("congestion_score")))]
        metric_rows = _metrics(metrics)
    else:
        rows = [payload]

    return UiResult(heading=heading, cards=tuple(cards), table_rows=tuple(rows), metric_rows=metric_rows, source_mode=str(payload.get("data_mode") or "UNKNOWN SOURCE MODE"), confidence=_confidence(payload), analysis_period=_period(payload.get("period")), sources=_sources(payload), assumptions=_assumptions(payload))


def build_model_status(health: Any) -> ModelStatusView:
    message, ready, enabled = str(getattr(health, "message", "Status unavailable.")), bool(getattr(health, "ready", False)), bool(getattr(health, "enabled", False))
    state = str(getattr(health, "state", "unknown"))
    if ready:
        return ModelStatusView(label="LM Studio ready", detail=message, severity="success")
    if not enabled:
        return ModelStatusView(label="Deterministic mode", detail=message, severity="info")
    if state in {"timeout", "unavailable", "invalid_response"}:
        return ModelStatusView(label="LM Studio unavailable", detail=message + " Required demo questions still use deterministic fallback.", severity="warning")
    return ModelStatusView(label="LM Studio running", detail=message, severity="info")
