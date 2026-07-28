import pytest

from src.agent import RouteSource, ToolExecutionResult
from src.explanations import ExplanationGenerator, ExplanationSource


class FakeClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.response_text}}]}


def execution(tool: str = "get_airport_profile") -> ToolExecutionResult:
    return ToolExecutionResult(
        question="secret original question",
        tool_name=tool,
        arguments={"airport_code": "SFO"},
        route_source=RouteSource.FALLBACK,
        output={
            "data_mode": "ILLUSTRATIVE DEMO DATA",
            "period": {
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "label": "Calendar year 2025",
            },
            "sources": [
                {
                    "source_name": "Illustrative airport fixture",
                    "data_mode": "ILLUSTRATIVE DEMO DATA",
                    "retrieved_at": "2026-07-27T00:00:00Z",
                    "period": {
                        "start_date": "2025-01-01",
                        "end_date": "2025-12-31",
                        "label": "Calendar year 2025",
                    },
                }
            ],
            "assumptions": ["Synthetic values only."],
            "analysis": {
                "airport": {
                    "airport_code": "SFO",
                    "name": "San Francisco International Airport",
                },
                "metrics": {
                    "congestion_score": 42,
                    "investment_opportunity_score": 55,
                },
                "confidence": {"level": "HIGH"},
                "assumptions": ["Screening proxy, not expected return."],
            },
        },
    )


@pytest.mark.parametrize(
    "unsupported_text",
    [
        "This airport guarantees profit.",
        "SFO is located in Europe.",
        "The expansion is risk-free.",
        "The opportunity score is 2025.",
        "The opportunity score is 5.5e1.",
        "The opportunity score is fifty-five.",
    ],
)
def test_arbitrary_model_prose_is_never_displayed(unsupported_text):
    client = FakeClient(unsupported_text)
    result = ExplanationGenerator(client).generate(execution())

    assert result.source is ExplanationSource.TEMPLATE
    assert client.calls == []
    assert unsupported_text not in result.text
    assert "Congestion score: 42" in result.text
    assert "Investment opportunity score: 55" in result.text


def test_deterministic_provenance_is_complete():
    result = ExplanationGenerator(None).generate(execution())

    assert "Source mode: ILLUSTRATIVE DEMO DATA" in result.text
    assert "Analysis period: 2025-01-01 to 2025-12-31" in result.text
    assert "Illustrative airport fixture" in result.text
    assert "retrieved/fixture date 2026-07-27" in result.text
    assert "source period 2025-01-01 to 2025-12-31" in result.text
    assert "Synthetic values only." in result.text
    assert "Screening proxy, not expected return." in result.text
    assert result.assumptions == (
        "Synthetic values only.",
        "Screening proxy, not expected return.",
    )


def test_original_question_is_not_rendered():
    result = ExplanationGenerator(None).generate(execution())
    assert "secret original question" not in result.text
