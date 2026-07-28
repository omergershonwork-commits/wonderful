"""Full Streamlit UI for the Airport Investment Intelligence Agent."""
from __future__ import annotations

import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from src.agent import (
    AgentError,
    AgentToolCaller,
    LMStudioChatClient,
    RoutingPolicy,
    ToolDispatcher,
)
from src.config import Settings, get_settings
from src.conversation import (
    ConversationManager,
    ConversationResolutionError,
    ConversationState,
)
from src.explanations import ExplanationGenerator
from src.llm_client import LMStudioError, LMStudioHealth
from src.router import (
    AirportQuestionRouter,
    DeterministicFallbackRouter,
    FallbackRoutingError,
)
from src.tools import AirportAnalyticsService
from src.ui import SUGGESTED_PROMPTS, UiResult, build_model_status, build_ui_result

DISCLAIMER = (
    "This application screens airport capacity-expansion signals using synthetic "
    "demonstration data and deterministic proxies. It does not estimate actual "
    "investment return or observed lost demand."
)


@dataclass(frozen=True, slots=True)
class Runtime:
    client: LMStudioChatClient
    conversation: ConversationManager
    explanations: ExplanationGenerator
    policy: RoutingPolicy


def policy_from_settings(settings: Settings) -> RoutingPolicy:
    return RoutingPolicy(
        ranking_limit=5,
        long_haul_threshold_miles=settings.long_haul_threshold_miles,
        target_load_factor=settings.target_load_factor,
    )


def _conversation_signature(
    policy: RoutingPolicy,
    min_annual_passengers: int | None,
) -> tuple[int, int, float, int | None]:
    return (
        policy.ranking_limit,
        policy.long_haul_threshold_miles,
        policy.target_load_factor,
        min_annual_passengers,
    )


def build_runtime(settings: Settings) -> Runtime:
    policy = policy_from_settings(settings)
    client = LMStudioChatClient.from_settings(settings)
    service = AirportAnalyticsService(
        min_annual_passengers=settings.min_annual_passengers
    )
    dispatcher = ToolDispatcher(service)
    agent = (
        AgentToolCaller(client, dispatcher=dispatcher, policy=policy)
        if settings.use_llm
        else None
    )
    fallback = DeterministicFallbackRouter(dispatcher, policy=policy)
    router = AirportQuestionRouter(agent, fallback=fallback)
    conversation = ConversationManager(router, dispatcher, policy=policy)
    return Runtime(client, conversation, ExplanationGenerator(client), policy)


def runtime_for_session(settings: Settings) -> Runtime:
    signature = (
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        settings.use_llm,
        settings.http_timeout_seconds,
        settings.long_haul_threshold_miles,
        settings.target_load_factor,
        settings.min_annual_passengers,
    )
    cached = st.session_state.get("_runtime")
    if cached is None or st.session_state.get("_runtime_signature") != signature:
        cached = build_runtime(settings)
        st.session_state["_runtime"] = cached
        st.session_state["_runtime_signature"] = signature
        st.session_state.pop("_lm_health", None)
    return cached


def synchronize_conversation_session(
    session_state: MutableMapping[str, Any],
    policy: RoutingPolicy,
    min_annual_passengers: int | None = None,
) -> bool:
    """Reset state and visible history whenever analytical policy changes."""

    signature = _conversation_signature(policy, min_annual_passengers)
    changed = session_state.get("_conversation_policy_signature") != signature
    if changed:
        session_state["conversation_state"] = ConversationState.from_policy(
            policy
        ).model_dump(mode="json")
        session_state["messages"] = []
        session_state["_conversation_policy_signature"] = signature
        return True
    session_state.setdefault(
        "conversation_state",
        ConversationState.from_policy(policy).model_dump(mode="json"),
    )
    session_state.setdefault("messages", [])
    return False


def initialize_session(
    policy: RoutingPolicy,
    min_annual_passengers: int | None = None,
) -> None:
    synchronize_conversation_session(
        st.session_state,
        policy,
        min_annual_passengers,
    )


def cached_health(runtime: Runtime, *, force: bool = False) -> LMStudioHealth:
    now = time.monotonic()
    cached: dict[str, Any] | None = st.session_state.get("_lm_health")
    if not force and cached and now - float(cached["checked_at"]) < 30:
        return LMStudioHealth.model_validate(cached["payload"])
    status = runtime.client.health_check()
    st.session_state["_lm_health"] = {
        "checked_at": now,
        "payload": status.model_dump(mode="json"),
    }
    return status


def render_status(runtime: Runtime) -> None:
    refresh = st.button("Refresh LM Studio status", use_container_width=True)
    status = build_model_status(cached_health(runtime, force=refresh))
    renderer = {"success": st.success, "warning": st.warning}.get(
        status.severity, st.info
    )
    renderer(f"**{status.label}**\n\n{status.detail}")


def render_view(view: UiResult) -> None:
    st.subheader(view.heading)
    if view.cards:
        columns = st.columns(min(4, len(view.cards)))
        for index, card in enumerate(view.cards):
            columns[index % len(columns)].metric(
                card.label, card.value, help=card.help_text
            )
    if view.table_rows:
        st.dataframe(
            pd.DataFrame(list(view.table_rows)),
            use_container_width=True,
            hide_index=True,
        )
    if view.metric_rows:
        st.markdown("#### Metric breakdown")
        st.dataframe(
            pd.DataFrame(list(view.metric_rows)),
            use_container_width=True,
            hide_index=True,
        )
    metadata = st.columns(3)
    metadata[0].metric("Source mode", view.source_mode)
    metadata[1].metric("Confidence", view.confidence)
    metadata[2].metric("Analysis period", view.analysis_period)
    with st.expander("Sources, assumptions, and limitations"):
        if view.sources:
            st.dataframe(
                pd.DataFrame(list(view.sources)),
                use_container_width=True,
                hide_index=True,
            )
        for item in view.assumptions or (
            "No additional assumptions were reported.",
        ):
            st.markdown(f"- {item}")


def methodology() -> None:
    with st.expander("Methodology"):
        st.markdown(
            """
- Passenger growth uses immediately adjacent equivalent periods.
- Congestion combines delay, taxi-out, cancellations, and departures per runway.
- Opportunity scoring combines growth, load factor, congestion, unmet-capacity proxy, and market scale.
- Missing inputs are renormalized and reduce confidence through one explicit uncertainty penalty.
- Long-haul routes include flights exactly at the configured mileage threshold.
- Unmet capacity is a screening proxy, not measured lost demand.
- Qwen may select one approved tool; Python validates intent and performs every calculation.
- User-visible explanations are deterministic templates.
"""
        )


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message.get("view"):
                render_view(UiResult.model_validate(message["view"]))


def suggested_prompt() -> str | None:
    st.markdown("#### Suggested questions")
    columns = st.columns(2)
    for index, prompt in enumerate(SUGGESTED_PROMPTS):
        if columns[index % 2].button(
            prompt,
            key=f"prompt-{index}",
            use_container_width=True,
        ):
            return prompt
    return None


def handle_prompt(prompt: str, runtime: Runtime) -> None:
    state = ConversationState.model_validate(st.session_state.conversation_state)
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Running deterministic airport analysis..."):
                turn = runtime.conversation.handle(prompt, state)
                explanation = runtime.explanations.generate(turn.execution)
                view = build_ui_result(turn.execution)
            st.markdown(explanation.text)
            render_view(view)
            st.session_state.conversation_state = turn.state.model_dump(mode="json")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": explanation.text,
                    "view": view.model_dump(mode="json"),
                }
            )
        except (
            ConversationResolutionError,
            FallbackRoutingError,
            AgentError,
            LMStudioError,
            ValueError,
        ) as exc:
            message = (
                "I could not map that request to a supported airport analysis. "
                f"Details: {exc}"
            )
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "text": message}
            )


def render_app() -> None:
    st.set_page_config(
        page_title="Airport Investment Intelligence Agent",
        page_icon="✈️",
        layout="wide",
    )
    settings = get_settings()
    runtime = runtime_for_session(settings)
    initialize_session(runtime.policy, settings.min_annual_passengers)

    st.title("Airport Investment Intelligence Agent")
    st.caption(
        "Deterministic airport-capacity screening with optional local Qwen tool selection"
    )
    with st.sidebar:
        st.header("Runtime status")
        render_status(runtime)
        st.write(f"LM Studio endpoint: `{settings.llm_base_url}`")
        st.write(f"Configured model: `{settings.llm_model or 'not set'}`")
        st.write(
            "Long-haul threshold: "
            f"**{runtime.policy.long_haul_threshold_miles:,} miles**"
        )
        st.write(
            f"Target load factor: **{runtime.policy.target_load_factor:.0%}**"
        )
        st.write(
            "Minimum annual passengers: "
            f"**{settings.min_annual_passengers:,}**"
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.conversation_state = ConversationState.from_policy(
                runtime.policy
            ).model_dump(mode="json")
            st.session_state.messages = []
            st.rerun()

    if not st.session_state.messages:
        st.info(
            "Ask a suggested question, then continue with follow-ups such as "
            "‘Exclude Boston’, ‘What about passenger growth?’, or ‘Use 85%’."
        )
    suggested = suggested_prompt()
    render_history()
    typed = st.chat_input(
        "Ask an airport investment-intelligence question",
        key="follow-up-input",
    )
    prompt = suggested if suggested is not None else typed
    if prompt:
        handle_prompt(prompt, runtime)
    methodology()
    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    render_app()
