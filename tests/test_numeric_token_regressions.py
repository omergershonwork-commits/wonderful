from __future__ import annotations

import pytest

from src.agent import RoutingPolicy, ToolDispatcher
from src.conversation import (
    ConversationManager,
    ConversationResolutionError,
    ConversationState,
)
from src.numeric_tokens import (
    NumericTokenError,
    parse_canonical_decimal,
    parse_canonical_integer,
)
from src.router import (
    AirportQuestionRouter,
    DeterministicFallbackRouter,
    FallbackRoutingError,
)
from src.tools import AirportAnalyticsService


@pytest.fixture
def runtime():
    policy = RoutingPolicy()
    dispatcher = ToolDispatcher(AirportAnalyticsService())
    fallback = DeterministicFallbackRouter(dispatcher, policy=policy)
    router = AirportQuestionRouter(None, fallback=fallback)
    manager = ConversationManager(router, dispatcher, policy=policy)
    return router, manager


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("2500", 2500),
        ("+2500", 2500),
        ("-2,500", -2500),
        ("2,500", 2500),
    ],
)
def test_canonical_integer_tokens(token, expected):
    assert parse_canonical_integer(token, field_name="value") == expected


@pytest.mark.parametrize(
    "token",
    [
        "3.5",
        "3e1",
        "3_5",
        "3-5",
        "3,5",
        "2_500",
        "2-500",
        "2e3",
        "2,50",
    ],
)
def test_noncanonical_integer_tokens(token):
    with pytest.raises(NumericTokenError):
        parse_canonical_integer(token, field_name="value")


@pytest.mark.parametrize(
    "token",
    ["1e2", "0.8e1", "85,5", "85_5", "85e-1"],
)
def test_noncanonical_decimal_tokens(token):
    with pytest.raises(NumericTokenError):
        parse_canonical_decimal(token, field_name="value")


@pytest.mark.parametrize(
    "question",
    [
        "Rank the top 3e1 New England airports",
        "Rank the top 3_5 New England airports",
        "Rank the top 3-5 New England airports",
        "Rank 3,5 New England airports",
        "What share of ANC flights are long haul using 2_500 miles?",
        "What share of ANC flights are long haul using 2-500 miles?",
        "What share of ANC flights are long haul using 2e3 miles?",
        "What share of ANC flights are long haul using 2,50 miles?",
        "Estimate SFO unmet capacity at target load factor 1e2",
        "Estimate SFO unmet capacity at target load factor 0.8e1",
    ],
)
def test_initial_questions_reject_unsupported_complete_tokens(runtime, question):
    router, _ = runtime
    with pytest.raises(FallbackRoutingError):
        router.route(question)


@pytest.mark.parametrize(
    ("question", "state"),
    [
        ("Use 2_500 miles", ConversationState(airport_codes=("ANC",))),
        ("Use 2-500 miles", ConversationState(airport_codes=("ANC",))),
        ("What about top 3e1?", ConversationState(region="New England")),
        ("What about top 3_5?", ConversationState(region="New England")),
        ("Use 85,5%", ConversationState(airport_codes=("SFO",))),
        ("Use 85_5%", ConversationState(airport_codes=("SFO",))),
        ("Use 85e-1%", ConversationState(airport_codes=("SFO",))),
    ],
)
def test_followups_reject_unsupported_complete_tokens(runtime, question, state):
    _, manager = runtime
    with pytest.raises(ConversationResolutionError):
        manager.handle(question, state)
