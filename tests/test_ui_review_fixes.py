from types import SimpleNamespace

from app import synchronize_conversation_session
from src.agent import RoutingPolicy
from src.ui import build_ui_result


def test_comparison_ratios_are_formatted_as_percentages():
    output = {
        "data_mode": "ILLUSTRATIVE DEMO DATA",
        "period": {"start_date": "2025-01-01", "end_date": "2025-12-31"},
        "sources": [],
        "airports": [
            {
                "airport": {"airport_code": "LAX", "name": "Los Angeles"},
                "metrics": {
                    "passenger_growth": 0.08,
                    "load_factor": 0.9,
                    "congestion_score": 55.123,
                    "missing_components": [],
                },
                "confidence": {"level": "HIGH"},
            }
        ],
    }
    view = build_ui_result(
        SimpleNamespace(tool_name="compare_airports", output=output)
    )
    row = view.table_rows[0]
    assert row["Passenger Growth"] == "8.00%"
    assert row["Load Factor"] == "90.00%"
    assert row["Congestion Score"] == "55.12"


def test_policy_change_resets_conversation_state_and_history():
    state = {}
    first = RoutingPolicy(
        ranking_limit=5,
        long_haul_threshold_miles=3000,
        target_load_factor=0.82,
    )
    assert synchronize_conversation_session(state, first, 100_000) is True
    state["conversation_state"]["airport_codes"] = ["SFO"]
    state["messages"].append({"role": "user", "text": "Use 85%"})

    changed = RoutingPolicy(
        ranking_limit=4,
        long_haul_threshold_miles=2500,
        target_load_factor=0.85,
    )
    assert synchronize_conversation_session(state, changed, 100_000) is True
    assert state["messages"] == []
    assert state["conversation_state"]["airport_codes"] == []
    assert state["conversation_state"]["ranking_limit"] == 4
    assert state["conversation_state"]["long_haul_threshold_miles"] == 2500
    assert state["conversation_state"]["target_load_factor"] == 0.85

    assert synchronize_conversation_session(state, changed, 100_000) is False


def test_passenger_floor_change_resets_visible_ranking_history():
    state = {}
    policy = RoutingPolicy()
    assert synchronize_conversation_session(state, policy, 100_000) is True
    state["messages"].append({"role": "assistant", "text": "Old ranking"})
    state["conversation_state"]["region"] = "New England"

    assert synchronize_conversation_session(state, policy, 2_000_000) is True
    assert state["messages"] == []
    assert state["conversation_state"]["region"] is None
    assert synchronize_conversation_session(state, policy, 2_000_000) is False
