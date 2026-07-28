from types import SimpleNamespace

import pytest

from src.conversation import (
    ConversationManager,
    ConversationResolutionError,
    ConversationState,
    extract_airport_codes,
)


class FakeDispatcher:
    def __init__(self):
        self.calls = []

    def execute(self, *, question, tool_name, arguments, route_source=None):
        self.calls.append((tool_name, arguments))
        return SimpleNamespace(
            question=question,
            tool_name=tool_name,
            arguments=arguments,
            output={"tool": tool_name},
            route_source=route_source,
        )


class FakeRouter:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.calls = []

    def route(self, question):
        self.calls.append(question)
        lowered = question.casefold()
        if "new england" in lowered:
            return self.dispatcher.execute(
                question=question,
                tool_name="rank_airports",
                arguments={
                    "region": "New England",
                    "limit": 5,
                    "excluded_airports": [],
                },
            )
        if "sfo" in lowered and "profile" in lowered:
            return self.dispatcher.execute(
                question=question,
                tool_name="get_airport_profile",
                arguments={"airport_code": "SFO"},
            )
        if "sfo" in lowered:
            return self.dispatcher.execute(
                question=question,
                tool_name="estimate_unmet_capacity",
                arguments={"airport_code": "SFO", "target_load_factor": 0.82},
            )
        return self.dispatcher.execute(
            question=question,
            tool_name="compare_airports",
            arguments={
                "airport_codes": ["LAX", "SNA"],
                "metrics": ["congestion_score"],
            },
        )


@pytest.fixture
def manager():
    dispatcher = FakeDispatcher()
    return ConversationManager(FakeRouter(dispatcher), dispatcher), dispatcher


def test_airport_extraction_uses_complete_tokens():
    assert extract_airport_codes("Compare LAX and Santa Ana") == ("LAX", "SNA")
    assert extract_airport_codes("relax financial transform") == ()


def test_lax_and_sna_persist_across_metric_follow_up(manager):
    service, dispatcher = manager
    first = service.handle("Compare LAX and SNA congestion")
    second = service.handle("What about passenger growth?", first.state)
    assert first.state.airport_codes == ("LAX", "SNA")
    assert second.used_context is True
    assert second.execution.arguments == {
        "airport_codes": ["LAX", "SNA"],
        "metrics": ["passenger_growth"],
    }
    assert dispatcher.calls[-1][0] == "compare_airports"


def test_new_england_and_exclusions_persist(manager):
    service, _ = manager
    first = service.handle(
        "Which airports in New England are candidates for terminal expansion?"
    )
    second = service.handle("Exclude Boston", first.state)
    assert first.state.region == "New England"
    assert second.execution.arguments["region"] == "New England"
    assert second.execution.arguments["excluded_airports"] == ["BOS"]
    assert second.state.excluded_airports == ("BOS",)


def test_modified_target_load_factor_uses_previous_airport(manager):
    service, _ = manager
    first = service.handle("What is the unmet capacity in SFO?")
    second = service.handle("Use 85%", first.state)
    assert second.execution.tool_name == "estimate_unmet_capacity"
    assert second.execution.arguments == {
        "airport_code": "SFO",
        "target_load_factor": 0.85,
    }
    assert second.state.target_load_factor == 0.85


def test_follow_up_comparison_combines_previous_and_new_airport(manager):
    service, _ = manager
    first = service.handle("Show me the profile for SFO")
    second = service.handle("Compare it with ANC", first.state)
    assert second.execution.arguments["airport_codes"] == ["SFO", "ANC"]


def test_combined_follow_up_applies_new_airport_and_metric(manager):
    service, _ = manager
    first = service.handle("Show me the profile for SFO")
    second = service.handle(
        "Compare it with ANC using passenger growth", first.state
    )
    assert second.used_context is True
    assert second.execution.arguments == {
        "airport_codes": ["SFO", "ANC"],
        "metrics": ["passenger_growth"],
    }


def test_new_airport_question_replaces_old_airport_context(manager):
    service, dispatcher = manager
    first = service.handle("Compare LAX and SNA congestion")
    second = service.handle("Show me the profile for SFO", first.state)
    assert second.used_context is False
    assert second.state.airport_codes == ("SFO",)
    assert second.state.region is None
    assert dispatcher.calls[-1][0] == "get_airport_profile"


def test_exclusion_without_previous_region_fails_closed(manager):
    service, _ = manager
    with pytest.raises(ConversationResolutionError, match="regional ranking"):
        service.handle("Exclude Boston", ConversationState())
