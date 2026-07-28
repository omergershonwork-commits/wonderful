from types import SimpleNamespace

from src.ui import SUGGESTED_PROMPTS, build_model_status, build_ui_result


def execution(tool_name, output):
    return SimpleNamespace(tool_name=tool_name, output=output)


def provenance():
    return {
        "data_mode": "ILLUSTRATIVE DEMO DATA",
        "period": {
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "label": "Calendar year 2025",
        },
        "sources": [
            {
                "source_name": "Demo fixture manifest",
                "data_mode": "ILLUSTRATIVE DEMO DATA",
                "retrieved_at": "2026-07-27T00:00:00Z",
                "period": {
                    "start_date": "2025-01-01",
                    "end_date": "2025-12-31",
                },
            }
        ],
        "assumptions": ["Synthetic values only."],
    }


def test_rank_view_contains_table_score_and_provenance():
    payload = provenance()
    payload["results"] = [
        {
            "rank": 1,
            "recommendation": "Potential candidate",
            "analysis": {
                "airport": {
                    "airport_code": "BOS",
                    "name": "Boston Logan",
                    "state_code": "MA",
                },
                "metrics": {
                    "investment_opportunity_score": 72.5,
                    "congestion_score": 61.2,
                },
                "confidence": {"level": "HIGH"},
            },
        }
    ]
    view = build_ui_result(execution("rank_airports", payload))
    assert view.heading == "Ranked airport expansion candidates"
    assert view.cards[0].value == "1"
    assert view.table_rows[0]["Airport"] == "BOS"
    assert view.source_mode == "ILLUSTRATIVE DEMO DATA"
    assert view.analysis_period == "Calendar year 2025"
    assert view.sources[0]["Source"] == "Demo fixture manifest"


def test_long_haul_view_has_score_cards_and_routes():
    payload = provenance()
    payload.update(
        {
            "departure_share": 0.2836,
            "passenger_share": 0.3105,
            "long_haul_departures": 3800,
            "all_departures": 13400,
            "qualifying_routes": [
                {"destination_airport_code": "JFK", "departures": 1200}
            ],
            "confidence": {"level": "HIGH"},
        }
    )
    view = build_ui_result(execution("calculate_long_haul_share", payload))
    assert view.cards[0].value == "28.36%"
    assert view.cards[2].value == "3,800"
    assert view.table_rows[0]["destination_airport_code"] == "JFK"
    assert view.confidence == "HIGH"


def test_unmet_capacity_view_has_breakdown_and_disclaimer_help():
    payload = provenance()
    payload.update(
        {
            "breakdown": {
                "current_passengers": 54_000_000,
                "projected_passengers": 58_320_000,
                "target_load_factor": 0.82,
                "estimated_unmet_capacity_proxy": 8_121_951.2,
            },
            "confidence": {"level": "MEDIUM"},
        }
    )
    view = build_ui_result(execution("estimate_unmet_capacity", payload))
    assert view.cards[0].value == "8,121,951.20"
    assert "not observed lost demand" in view.cards[0].help_text
    assert any(row["Metric"] == "Target Load Factor" for row in view.metric_rows)


def test_model_status_and_suggested_prompts_cover_ui_requirements():
    ready = build_model_status(
        SimpleNamespace(
            state="ready",
            ready=True,
            enabled=True,
            message="Qwen is available.",
        )
    )
    offline = build_model_status(
        SimpleNamespace(
            state="unavailable",
            ready=False,
            enabled=True,
            message="Connection failed.",
        )
    )
    assert ready.severity == "success"
    assert "deterministic fallback" in offline.detail
    assert len(SUGGESTED_PROMPTS) == 4
