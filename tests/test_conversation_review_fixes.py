from types import SimpleNamespace

import pytest

from src.agent import RoutingPolicy, ToolDispatcher
from src.conversation import (
    ConversationManager,
    ConversationResolutionError,
    ConversationState,
)
from src.models import ComparisonMetric


class Service:
    def rank_airports(self, request):
        return {"x": 1}

    def compare_airports(self, request):
        return {"x": 1}

    def calculate_long_haul_share(self, request):
        return {"x": 1}

    def estimate_unmet_capacity(self, request):
        return {"x": 1}

    def get_airport_profile(self, request):
        return {"x": 1}


class Router:
    def route(self, question):
        if "long haul" in question.lower():
            return SimpleNamespace(
                tool_name="calculate_long_haul_share",
                arguments={"airport_code": "ANC", "threshold_miles": 3000},
                output={},
            )
        if "compare" in question.lower():
            return SimpleNamespace(
                tool_name="compare_airports",
                arguments={"airport_codes": ["LAX", "SNA"]},
                output={},
            )
        return SimpleNamespace(
            tool_name="get_airport_profile",
            arguments={"airport_code": "SFO"},
            output={},
        )


def manager(policy=None):
    return ConversationManager(Router(), ToolDispatcher(Service()), policy=policy)


def test_long_haul_does_not_store_invalid_comparison_metric():
    turn = manager().handle("What share of ANC flights are long haul?")
    assert turn.state.metric is None


def test_ambiguous_singular_pronoun_fails_closed():
    state = ConversationState(airport_codes=("LAX", "SNA"))
    with pytest.raises(ConversationResolutionError, match="ambiguous"):
        manager().handle("Compare it with ANC", state)


def test_single_airport_pronoun_is_resolved():
    state = ConversationState(airport_codes=("SFO",))
    turn = manager().handle("Compare it with ANC", state)
    assert turn.execution.arguments["airport_codes"] == ["SFO", "ANC"]


def test_combined_airport_and_metric_follow_up_is_resolved():
    state = ConversationState(airport_codes=("SFO",))
    turn = manager().handle("Compare it with ANC using passenger growth", state)
    assert turn.execution.arguments == {
        "airport_codes": ["SFO", "ANC"],
        "metrics": ["passenger_growth"],
    }


def test_new_comparison_without_selector_clears_stale_metric():
    state = ConversationState(
        airport_codes=("LAX", "SNA"), metric=ComparisonMetric.PASSENGER_GROWTH
    )
    turn = manager().handle("Compare them", state)
    assert turn.execution.arguments.get("metrics") is None
    assert turn.state.metric is None


def test_mixed_include_exclude_fails_closed():
    state = ConversationState(region="New England")
    with pytest.raises(ConversationResolutionError, match="include and exclude"):
        manager().handle("Include and exclude Boston", state)


def test_policy_initializes_state_defaults():
    policy = RoutingPolicy(
        ranking_limit=4,
        long_haul_threshold_miles=2700,
        target_load_factor=0.87,
    )
    state = manager(policy).initial_state()
    assert state.ranking_limit == 4
    assert state.long_haul_threshold_miles == 2700
    assert state.target_load_factor == 0.87


@pytest.mark.parametrize("question", ["Use -2500 miles", "Use -2,500 miles"])
def test_negative_mileage_follow_up_fails_closed(question):
    state = ConversationState(airport_codes=("ANC",))
    with pytest.raises(ConversationResolutionError, match="greater than zero"):
        manager().handle(question, state)


def test_negative_percentage_follow_up_fails_closed():
    state = ConversationState(airport_codes=("SFO",))
    with pytest.raises(ConversationResolutionError, match="greater than 0"):
        manager().handle("Use -5%", state)

@pytest.mark.parametrize(
    "question",
    [
        "Use 2,500.5 miles",
        "Use 2500.5 miles",
    ],
)
def test_fractional_contextual_mileage_fails_closed(question):
    state = ConversationState(airport_codes=("ANC",))
    with pytest.raises(ConversationResolutionError, match="whole number"):
        manager().handle(question, state)


def test_fractional_contextual_ranking_limit_fails_closed():
    state = ConversationState(region="New England")
    with pytest.raises(ConversationResolutionError, match="whole number"):
        manager().handle("What about top 3.5?", state)

@pytest.mark.parametrize(
    "question",
    [
        "What about top 3e1?",
        "What about top 3_5?",
        "What about top 3,5?",
    ],
)
def test_noncanonical_contextual_ranking_limit_fails_closed(question):
    state = ConversationState(region="New England")
    with pytest.raises(ConversationResolutionError, match="canonical whole number"):
        manager().handle(question, state)


@pytest.mark.parametrize(
    "question",
    [
        "Use 2_500 miles",
        "Use 2e3 miles",
        "Use 2,50 miles",
    ],
)
def test_noncanonical_contextual_mileage_fails_closed(question):
    state = ConversationState(airport_codes=("ANC",))
    with pytest.raises(ConversationResolutionError, match="canonical whole number"):
        manager().handle(question, state)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Use 2500 miles", 2500),
        ("Use 2,500 miles", 2500),
        ("Use +2500 miles", 2500),
    ],
)
def test_canonical_contextual_mileage_is_preserved(question, expected):
    state = ConversationState(airport_codes=("ANC",))
    turn = manager().handle(question, state)
    assert turn.execution.arguments["threshold_miles"] == expected

