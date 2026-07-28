import pytest

from src.agent import (
    RouteSource,
    ToolDispatcher,
    ToolExecutionError,
    ToolSelectionError,
)
from src.llm_client import LMStudioConnectionError
from src.router import (
    AirportQuestionRouter,
    DeterministicFallbackRouter,
    FallbackRoutingError,
)


class FakeService:
    def rank_airports(self, request):
        return {"tool": "rank_airports", "args": request.model_dump(mode="json")}

    def compare_airports(self, request):
        return {"tool": "compare_airports", "args": request.model_dump(mode="json")}

    def calculate_long_haul_share(self, request):
        return {"tool": "calculate_long_haul_share", "args": request.model_dump(mode="json")}

    def estimate_unmet_capacity(self, request):
        return {"tool": "estimate_unmet_capacity", "args": request.model_dump(mode="json")}

    def get_airport_profile(self, request):
        return {"tool": "get_airport_profile", "args": request.model_dump(mode="json")}


class FailingAgent:
    def __init__(self, exc, dispatcher):
        self.exc = exc
        self.dispatcher = dispatcher

    def route(self, question):
        raise self.exc


@pytest.fixture
def dispatcher():
    return ToolDispatcher(FakeService())


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        (
            "Which airports in New England are strong candidates for terminal expansion?",
            "rank_airports",
        ),
        ("Compare LA and Santa Ana airport congestion levels.", "compare_airports"),
        (
            "What percentage of long haul flights leave Anchorage?",
            "calculate_long_haul_share",
        ),
        ("What is the unmet flight demand in SFO and why?", "estimate_unmet_capacity"),
    ],
)
def test_required_questions_work_without_lm_studio(dispatcher, question, tool):
    result = AirportQuestionRouter(
        None,
        fallback=DeterministicFallbackRouter(dispatcher),
    ).route(question)
    assert result.tool_name == tool
    assert result.route_source is RouteSource.FALLBACK


def test_model_connection_failure_activates_fallback(dispatcher):
    agent = FailingAgent(LMStudioConnectionError("offline"), dispatcher)
    result = AirportQuestionRouter(agent).route("Compare LAX versus SNA congestion")
    assert result.tool_name == "compare_airports"
    assert result.route_source is RouteSource.FALLBACK


def test_model_execution_failure_activates_fallback(dispatcher):
    agent = FailingAgent(ToolExecutionError("unsupported XYZ"), dispatcher)
    result = AirportQuestionRouter(agent).route("What is SFO unmet capacity?")
    assert result.tool_name == "estimate_unmet_capacity"
    assert result.route_source is RouteSource.FALLBACK


def test_ordinary_prose_failure_activates_fallback(dispatcher):
    agent = FailingAgent(ToolSelectionError("prose"), dispatcher)
    result = AirportQuestionRouter(agent).route("What is SFO unmet capacity?")
    assert result.tool_name == "estimate_unmet_capacity"


@pytest.mark.parametrize(
    "question",
    [
        "Compare relax and Santa Ana congestion.",
        "What percentage of long haul financial activity occurred?",
        "What is transform capacity?",
        "Tell me a joke",
    ],
)
def test_substrings_and_unknown_questions_fail_closed(dispatcher, question):
    with pytest.raises(FallbackRoutingError):
        AirportQuestionRouter(
            None,
            fallback=DeterministicFallbackRouter(dispatcher),
        ).route(question)
