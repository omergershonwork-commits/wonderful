import json

import pytest

from src.agent import AgentToolCaller, RoutingPolicy, ToolArgumentsError, ToolDispatcher


class Client:
    def __init__(self, name, args):
        self.name, self.args = name, args

    def create_chat_completion(self, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": self.name,
                                    "arguments": json.dumps(self.args),
                                }
                            }
                        ]
                    }
                }
            ]
        }


class Service:
    def rank_airports(self, request):
        return request

    def compare_airports(self, request):
        return request

    def calculate_long_haul_share(self, request):
        return request

    def estimate_unmet_capacity(self, request):
        return request

    def get_airport_profile(self, request):
        return request


def route(name, args, question, policy=None):
    return AgentToolCaller(
        Client(name, args),
        dispatcher=ToolDispatcher(Service()),
        policy=policy,
    ).route(question)


def test_requested_metric_cannot_be_omitted():
    with pytest.raises(ToolArgumentsError, match="metric selector"):
        route(
            "compare_airports",
            {"airport_codes": ["LAX", "SNA"]},
            "Compare LAX and SNA congestion",
        )


def test_requested_exclusion_cannot_be_omitted():
    with pytest.raises(ToolArgumentsError, match="exclusions"):
        route(
            "rank_airports",
            {"region": "New England", "limit": 5},
            "Rank New England airports excluding Boston",
        )


@pytest.mark.parametrize("excluded", [["PVD"], ["BOS", "PVD"]])
def test_wrong_or_extra_exclusion_is_rejected(excluded):
    with pytest.raises(ToolArgumentsError, match="exclusions"):
        route(
            "rank_airports",
            {
                "region": "New England",
                "limit": 5,
                "excluded_airports": excluded,
            },
            "Rank New England airports excluding Boston",
        )


def test_exact_exclusion_and_metric_are_preserved():
    ranked = route(
        "rank_airports",
        {
            "region": "New England",
            "limit": 5,
            "excluded_airports": ["BOS"],
        },
        "Rank New England airports excluding Boston",
    )
    compared = route(
        "compare_airports",
        {
            "airport_codes": ["LAX", "SNA"],
            "metrics": ["congestion_score"],
        },
        "Compare LAX and SNA congestion",
    )
    assert ranked.arguments["excluded_airports"] == ["BOS"]
    assert compared.arguments["metrics"] == ["congestion_score"]


def test_exclusion_actions_apply_only_to_airports_in_their_scope():
    result = route(
        "rank_airports",
        {
            "region": "New England",
            "limit": 5,
            "excluded_airports": ["BOS"],
        },
        "Rank New England airports excluding Boston but include Providence",
    )
    assert result.arguments["excluded_airports"] == ["BOS"]


@pytest.mark.parametrize(
    ("question", "excluded"),
    [
        ("Rank New England airports exclude Boston but include Boston", []),
        ("Rank New England airports include Boston but exclude Boston", ["BOS"]),
    ],
)
def test_repeated_actions_on_the_same_airport_are_applied_in_order(question, excluded):
    result = route(
        "rank_airports",
        {
            "region": "New England",
            "limit": 5,
            "excluded_airports": excluded,
        },
        question,
    )
    assert result.arguments["excluded_airports"] == excluded


def test_runtime_policy_supplies_omitted_numeric_defaults():
    policy = RoutingPolicy(
        ranking_limit=4,
        long_haul_threshold_miles=2800,
        target_load_factor=0.86,
    )
    result = route(
        "calculate_long_haul_share",
        {"airport_code": "ANC"},
        "What share of ANC flights are long haul?",
        policy,
    )
    assert result.arguments["threshold_miles"] == 2800


@pytest.mark.parametrize(
    ("tool_name", "arguments", "question"),
    [
        (
            "rank_airports",
            {"region": "New England", "excluded_airports": []},
            "Rank the top 0 New England airports",
        ),
        (
            "calculate_long_haul_share",
            {"airport_code": "ANC"},
            "What share of ANC flights are long haul using 0 miles?",
        ),
        (
            "estimate_unmet_capacity",
            {"airport_code": "SFO"},
            "Estimate SFO unmet capacity at a target load factor of 0%",
        ),
    ],
)
def test_explicit_zero_values_are_not_replaced_by_defaults(
    tool_name, arguments, question
):
    with pytest.raises(ToolArgumentsError, match="failed validation"):
        route(tool_name, arguments, question)


@pytest.mark.parametrize(
    ("tool_name", "arguments", "question"),
    [
        (
            "rank_airports",
            {"region": "New England", "excluded_airports": []},
            "Rank the top -3 New England airports",
        ),
        (
            "calculate_long_haul_share",
            {"airport_code": "ANC"},
            "What share of ANC flights are long haul using -2500 miles?",
        ),
        (
            "estimate_unmet_capacity",
            {"airport_code": "SFO"},
            "Estimate SFO unmet capacity at a target load factor of -85%",
        ),
    ],
)
def test_negative_overrides_are_not_reinterpreted_as_positive(
    tool_name, arguments, question
):
    with pytest.raises(ToolArgumentsError, match="failed validation"):
        route(tool_name, arguments, question)


def test_bare_load_factor_above_one_is_not_guessed_as_a_percentage():
    with pytest.raises(ToolArgumentsError, match="failed validation"):
        route(
            "estimate_unmet_capacity",
            {"airport_code": "SFO"},
            "Estimate SFO unmet capacity at a target load factor of 1.5",
        )

@pytest.mark.parametrize(
    ("tool_name", "arguments", "question"),
    [
        (
            "calculate_long_haul_share",
            {"airport_code": "ANC", "threshold_miles": 5},
            "What share of ANC flights are long haul using 2,500.5 miles?",
        ),
        (
            "calculate_long_haul_share",
            {"airport_code": "ANC", "threshold_miles": 5},
            "What share of ANC flights are long haul using 2500.5 miles?",
        ),
        (
            "rank_airports",
            {"region": "New England", "limit": 3, "excluded_airports": []},
            "Rank the top 3.5 New England airports",
        ),
    ],
)
def test_fractional_values_in_integer_only_fields_fail_closed(
    tool_name, arguments, question
):
    with pytest.raises(ToolArgumentsError, match="whole numbers"):
        route(tool_name, arguments, question)

@pytest.mark.parametrize(
    ("tool_name", "arguments", "question"),
    [
        (
            "rank_airports",
            {"region": "New England", "limit": 3, "excluded_airports": []},
            "Rank the top 3e1 New England airports",
        ),
        (
            "rank_airports",
            {"region": "New England", "limit": 3, "excluded_airports": []},
            "Rank the top 3_5 New England airports",
        ),
        (
            "rank_airports",
            {"region": "New England", "limit": 5, "excluded_airports": []},
            "Rank 3,5 New England airports",
        ),
        (
            "calculate_long_haul_share",
            {"airport_code": "ANC", "threshold_miles": 500},
            "What share of ANC flights are long haul using 2_500 miles?",
        ),
        (
            "calculate_long_haul_share",
            {"airport_code": "ANC"},
            "What share of ANC flights are long haul using 2e3 miles?",
        ),
        (
            "calculate_long_haul_share",
            {"airport_code": "ANC"},
            "What share of ANC flights are long haul using 2,50 miles?",
        ),
    ],
)
def test_noncanonical_integer_tokens_fail_closed(tool_name, arguments, question):
    with pytest.raises(ToolArgumentsError, match="canonical whole number"):
        route(tool_name, arguments, question)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What share of ANC flights are long haul using 2500 miles?", 2500),
        ("What share of ANC flights are long haul using 2,500 miles?", 2500),
        ("What share of ANC flights are long haul using +2500 miles?", 2500),
    ],
)
def test_canonical_mileage_tokens_are_preserved(question, expected):
    result = route(
        "calculate_long_haul_share",
        {"airport_code": "ANC", "threshold_miles": expected},
        question,
    )
    assert result.arguments["threshold_miles"] == expected

