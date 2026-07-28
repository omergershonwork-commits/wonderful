"""End-to-end acceptance tests that require neither internet nor LM Studio."""

from __future__ import annotations

import io
import zipfile

import httpx
import pytest

from app import synchronize_conversation_session
from src.agent import (
    RouteSource,
    RoutingPolicy,
    ToolArgumentsError,
    ToolDispatcher,
)
from src.conversation import (
    ConversationManager,
    ConversationResolutionError,
    ConversationState,
)
from src.data.public_clients import (
    PublicDataInvalidResponseError,
    _optional_nonnegative,
    _parse_number,
)
from src.data.public_clients import (
    BtsOnTimeClient,
    BtsT100Client,
    CachedHttpClient,
    DiskResponseCache,
    PublicDataInvalidResponseError,
)
from src.models import ComparisonMetric
from src.router import (
    AirportQuestionRouter,
    DeterministicFallbackRouter,
    FallbackRoutingError,
)
from src.scoring import AirportScoringInput, score_airports
from src.tools import AirportAnalyticsService


@pytest.fixture
def offline_runtime():
    """Build the real fixture-backed runtime with the LLM path disabled."""

    policy = RoutingPolicy(
        ranking_limit=5,
        long_haul_threshold_miles=3000,
        target_load_factor=0.82,
    )
    dispatcher = ToolDispatcher(AirportAnalyticsService())
    fallback = DeterministicFallbackRouter(dispatcher, policy=policy)
    router = AirportQuestionRouter(None, fallback=fallback)
    return policy, dispatcher, router


def test_four_assignment_questions_work_without_network_or_lm_studio(
    offline_runtime, monkeypatch
):
    """Exercise the real repository, metrics, scoring, tools, and fallback router."""

    def reject_network(*args, **kwargs):
        raise AssertionError("offline acceptance tests must not perform HTTP requests")

    monkeypatch.setattr(httpx.Client, "send", reject_network)
    _, _, router = offline_runtime

    ranking = router.route(
        "Which airports in New England are strong candidates for terminal expansion?"
    )
    comparison = router.route("Compare LAX and SNA congestion levels.")
    long_haul = router.route(
        "What is the percentage of long haul flights out of ANC?"
    )
    capacity = router.route("What is the unmet flight demand in SFO and why?")

    assert ranking.route_source is RouteSource.FALLBACK
    assert ranking.tool_name == "rank_airports"
    assert len(ranking.output.results) == 5

    assert comparison.tool_name == "compare_airports"
    assert [item.airport.airport_code for item in comparison.output.airports] == [
        "LAX",
        "SNA",
    ]

    assert long_haul.tool_name == "calculate_long_haul_share"
    assert long_haul.output.all_departures == 13_400
    assert long_haul.output.long_haul_departures == 3_800

    assert capacity.tool_name == "estimate_unmet_capacity"
    assert capacity.output.breakdown.estimated_unmet_capacity_proxy >= 0


def test_explicit_overrides_survive_the_complete_offline_path(offline_runtime):
    _, _, router = offline_runtime

    ranking = router.route(
        "Rank the top 3 New England terminal expansion candidates."
    )
    long_haul = router.route(
        "What share of ANC flights are long haul using 2,500 miles?"
    )
    capacity = router.route(
        "Estimate SFO unmet capacity at a target load factor of 85%."
    )

    assert ranking.arguments["limit"] == 3
    assert len(ranking.output.results) == 3
    assert long_haul.arguments["threshold_miles"] == 2500
    assert long_haul.output.input.threshold_miles == 2500
    assert capacity.arguments["target_load_factor"] == pytest.approx(0.85)
    assert capacity.output.breakdown.target_load_factor == pytest.approx(0.85)


@pytest.mark.parametrize(
    "question",
    [
        "Rank the top 0 New England airports.",
        "Rank the top -3 New England airports.",
        "What share of ANC flights are long haul using 0 miles?",
        "What share of ANC flights are long haul using -2,500 miles?",
        "Estimate SFO unmet capacity at a target load factor of 0%.",
        "Estimate SFO unmet capacity at a target load factor of -85%.",
    ],
)
def test_invalid_signed_or_zero_overrides_fail_closed(
    offline_runtime,
    question,
):
    _, _, router = offline_runtime
    with pytest.raises(ToolArgumentsError):
        router.route(question)


def test_exclusion_actions_are_bound_to_the_named_airports(offline_runtime):
    _, _, router = offline_runtime
    result = router.route(
        "Rank New England airports excluding Boston but include Providence."
    )
    assert result.arguments["excluded_airports"] == ["BOS"]
    assert all(
        item.analysis.airport.airport_code != "BOS"
        for item in result.output.results
    )
    assert any(
        item.analysis.airport.airport_code == "PVD"
        for item in result.output.results
    )


def test_conversation_follow_ups_execute_against_real_tools(offline_runtime):
    policy, dispatcher, router = offline_runtime
    manager = ConversationManager(router, dispatcher, policy=policy)

    comparison = manager.handle("Compare LAX and SNA congestion.")
    growth = manager.handle("What about passenger growth?", comparison.state)

    assert growth.used_context is True
    assert growth.execution.arguments["airport_codes"] == ["LAX", "SNA"]
    assert growth.execution.arguments["metrics"] == [
        ComparisonMetric.PASSENGER_GROWTH.value
    ]

    ranking = manager.handle(
        "Which airports in New England are terminal expansion candidates?"
    )
    excluded = manager.handle("Exclude Boston", ranking.state)

    assert excluded.execution.arguments["excluded_airports"] == ["BOS"]
    assert all(
        item.analysis.airport.airport_code != "BOS"
        for item in excluded.execution.output.results
    )


def test_combined_airport_and_metric_follow_up_uses_real_tools(offline_runtime):
    policy, dispatcher, router = offline_runtime
    manager = ConversationManager(router, dispatcher, policy=policy)

    profile = manager.handle("Show me the profile for SFO.")
    comparison = manager.handle(
        "Compare it with ANC using passenger growth.",
        profile.state,
    )

    assert comparison.used_context is True
    assert comparison.execution.arguments == {
        "airport_codes": ["SFO", "ANC"],
        "metrics": [ComparisonMetric.PASSENGER_GROWTH.value],
    }


def test_runtime_policy_change_resets_existing_conversation_state():
    session_state = {}
    original = RoutingPolicy(
        ranking_limit=5,
        long_haul_threshold_miles=3000,
        target_load_factor=0.82,
    )
    synchronize_conversation_session(session_state, original)
    session_state["conversation_state"]["airport_codes"] = ["SFO"]
    session_state["messages"] = [{"role": "user", "text": "Use 85%"}]

    changed = RoutingPolicy(
        ranking_limit=4,
        long_haul_threshold_miles=2500,
        target_load_factor=0.85,
    )
    assert synchronize_conversation_session(session_state, changed) is True
    assert session_state["messages"] == []
    assert session_state["conversation_state"]["airport_codes"] == []
    assert session_state["conversation_state"]["ranking_limit"] == 4
    assert session_state["conversation_state"]["long_haul_threshold_miles"] == 2500
    assert session_state["conversation_state"]["target_load_factor"] == 0.85


def _public_http(tmp_path, handler):
    return CachedHttpClient(
        cache=DiskResponseCache(tmp_path / "cache", ttl_seconds=3600),
        timeout_seconds=2,
        max_download_bytes=10_000_000,
        transport=httpx.MockTransport(handler),
    )


def test_partial_t100_period_is_not_labelled_as_full_calendar_year(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "origin_airport_code": "SFO",
                    "year": "2026",
                    "reporting_month": "2026-08-01T00:00:00.000",
                    "total_departures": "120000",
                    "total_passengers": "35000000",
                    "total_seats": "42000000",
                }
            ],
            request=request,
        )

    result = BtsT100Client(_public_http(tmp_path, handler)).fetch_airport_year(
        "SFO", 2026
    )
    assert result.source.period.end_date.isoformat() == "2026-08-31"
    assert result.source.period.label == "Year-to-date through 2026-08"


def _on_time_zip(row: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "On_Time.csv",
            "Origin,Cancelled,DepDelay,TaxiOut\n" + row,
        )
    return buffer.getvalue()


@pytest.mark.parametrize(
    "row",
    [
        "SFO,2,12,18\n",
        "SFO,0,-3,18\n",
        "SFO,0,12,-1\n",
    ],
)
def test_invalid_on_time_values_are_rejected_in_acceptance_path(
    tmp_path,
    row,
):
    payload = _on_time_zip(row)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, request=request)

    client = BtsOnTimeClient(_public_http(tmp_path, handler))
    with pytest.raises(PublicDataInvalidResponseError):
        client.fetch_origin_month("SFO", 2026, 5)


def test_missing_congestion_evidence_is_penalized_once():
    incomplete = AirportScoringInput(
        airport_code="AAA",
        passengers=1_000_000,
        passenger_growth=0.05,
        load_factor=0.8,
        average_departure_delay_minutes=None,
        average_taxi_out_minutes=None,
        cancellation_rate=None,
        departures_per_runway=None,
        estimated_unmet_capacity_proxy=100_000,
    )
    complete = AirportScoringInput(
        airport_code="BBB",
        passengers=2_000_000,
        passenger_growth=0.06,
        load_factor=0.82,
        average_departure_delay_minutes=12,
        average_taxi_out_minutes=18,
        cancellation_rate=0.02,
        departures_per_runway=20_000,
        estimated_unmet_capacity_proxy=120_000,
    )

    result = score_airports([incomplete, complete])["AAA"]

    assert set(result.missing_components) == {
        "average_departure_delay_minutes",
        "average_taxi_out_minutes",
        "cancellation_rate",
        "departures_per_runway",
    }
    assert result.uncertainty_penalty == 16
    assert result.confidence.score == pytest.approx(0.5)


def test_repeated_actions_on_one_airport_preserve_final_instruction(offline_runtime):
    _, _, router = offline_runtime
    included = router.route(
        "Rank New England airports exclude Boston but include Boston"
    )
    excluded = router.route(
        "Rank New England airports include Boston but exclude Boston"
    )
    assert included.arguments["excluded_airports"] == []
    assert excluded.arguments["excluded_airports"] == ["BOS"]


def test_bare_decimal_load_factor_above_one_fails_closed(offline_runtime):
    _, _, router = offline_runtime
    with pytest.raises(ToolArgumentsError):
        router.route(
            "Estimate SFO unmet capacity at a target load factor of 1.5"
        )


@pytest.mark.parametrize("question", ["Use -2500 miles", "Use -2,500 miles"])
def test_signed_conversational_mileage_fails_closed(offline_runtime, question):
    policy, dispatcher, router = offline_runtime
    manager = ConversationManager(router, dispatcher, policy=policy)
    state = ConversationState(airport_codes=("ANC",))
    with pytest.raises(ConversationResolutionError):
        manager.handle(question, state)


def test_signed_conversational_percentage_fails_closed(offline_runtime):
    policy, dispatcher, router = offline_runtime
    manager = ConversationManager(router, dispatcher, policy=policy)
    state = ConversationState(airport_codes=("SFO",))
    with pytest.raises(ConversationResolutionError):
        manager.handle("Use -5%", state)


def test_passenger_floor_only_change_resets_visible_history():
    session = {}
    policy = RoutingPolicy()
    assert synchronize_conversation_session(session, policy, 100_000) is True
    session["messages"].append({"role": "assistant", "text": "Old ranking"})
    session["conversation_state"]["region"] = "New England"
    assert synchronize_conversation_session(session, policy, 2_000_000) is True
    assert session["messages"] == []
    assert session["conversation_state"]["region"] is None


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_public_values_fail_closed(value):
    with pytest.raises(PublicDataInvalidResponseError, match="non-finite"):
        _parse_number(value, field_name="test value")
    with pytest.raises(PublicDataInvalidResponseError, match="non-finite"):
        _optional_nonnegative(value, field_name="departure delay")

@pytest.mark.parametrize(
    "question",
    [
        "What share of ANC flights are long haul using 2,500.5 miles?",
        "What share of ANC flights are long haul using 2500.5 miles?",
        "Rank the top 3.5 New England airports.",
    ],
)
def test_fractional_integer_overrides_fail_in_complete_offline_path(
    offline_runtime,
    question,
):
    _, _, router = offline_runtime
    with pytest.raises(FallbackRoutingError):
        router.route(question)


@pytest.mark.parametrize(
    ("question", "state"),
    [
        ("Use 2,500.5 miles", ConversationState(airport_codes=("ANC",))),
        ("Use 2500.5 miles", ConversationState(airport_codes=("ANC",))),
        ("What about top 3.5?", ConversationState(region="New England")),
    ],
)
def test_fractional_contextual_values_fail_in_complete_offline_path(
    offline_runtime,
    question,
    state,
):
    policy, dispatcher, router = offline_runtime
    manager = ConversationManager(router, dispatcher, policy=policy)
    with pytest.raises(ConversationResolutionError, match="whole number"):
        manager.handle(question, state)

@pytest.mark.parametrize(
    "question",
    [
        "Rank the top 3e1 New England airports",
        "Rank the top 3_5 New England airports",
        "Rank 3,5 New England airports",
        "What share of ANC flights are long haul using 2_500 miles?",
        "What share of ANC flights are long haul using 2e3 miles?",
        "What share of ANC flights are long haul using 2,50 miles?",
    ],
)
def test_noncanonical_integer_tokens_fail_in_complete_offline_path(offline_runtime, question):
    _, _, router = offline_runtime
    with pytest.raises(FallbackRoutingError):
        router.route(question)


@pytest.mark.parametrize(
    ("question", "state"),
    [
        ("What about top 3e1?", ConversationState(region="New England")),
        ("What about top 3_5?", ConversationState(region="New England")),
        ("Use 2_500 miles", ConversationState(airport_codes=("ANC",))),
        ("Use 2e3 miles", ConversationState(airport_codes=("ANC",))),
        ("Use 2,50 miles", ConversationState(airport_codes=("ANC",))),
    ],
)
def test_noncanonical_contextual_tokens_fail_in_complete_offline_path(offline_runtime, question, state):
    policy, dispatcher, router = offline_runtime
    manager = ConversationManager(router, dispatcher, policy=policy)
    with pytest.raises(ConversationResolutionError, match="canonical whole number"):
        manager.handle(question, state)

