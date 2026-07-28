"""Deterministic fallback routing for supported airport questions."""

from __future__ import annotations

from src.agent import (
    AgentToolCaller,
    LMStudioModelNotConfiguredError,
    RouteSource,
    RoutingPolicy,
    SelectedToolCall,
    ToolArgumentsError,
    ToolDispatcher,
    ToolExecutionError,
    ToolExecutionResult,
    ToolSelectionError,
    UnknownToolError,
)
from src.llm_client import LMStudioError
from src.question_constraints import parse_question_constraints


class FallbackRoutingError(RuntimeError):
    """Raised when no supported deterministic route matches the question."""


class DeterministicFallbackRouter:
    """Build approved tool calls from the shared deterministic question parser."""

    def __init__(
        self,
        dispatcher: ToolDispatcher | None = None,
        *,
        policy: RoutingPolicy | None = None,
    ) -> None:
        self._dispatcher = dispatcher or ToolDispatcher()
        self._policy = policy or RoutingPolicy()

    @property
    def dispatcher(self) -> ToolDispatcher:
        return self._dispatcher

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    def select(self, question: str) -> SelectedToolCall:
        try:
            constraints = parse_question_constraints(question)
            return SelectedToolCall(
                name=constraints.expected_tool,
                arguments=constraints.expected_arguments(self._policy),
            )
        except (ToolArgumentsError, ValueError) as exc:
            raise FallbackRoutingError(
                "No deterministic fallback route matched the question"
            ) from exc

    def route(self, question: str) -> ToolExecutionResult:
        normalized = question.strip()
        if not normalized:
            raise FallbackRoutingError("question must not be empty")
        selected = self.select(normalized)
        return self._dispatcher.execute(
            question=normalized,
            tool_name=selected.name,
            arguments=selected.arguments,
            route_source=RouteSource.FALLBACK,
        )


class AirportQuestionRouter:
    """Prefer validated LM Studio routing and fall back deterministically."""

    def __init__(
        self,
        agent: AgentToolCaller | None,
        *,
        fallback: DeterministicFallbackRouter | None = None,
    ) -> None:
        self._agent = agent
        dispatcher = getattr(agent, "dispatcher", None) if agent is not None else None
        policy = getattr(agent, "policy", None) if agent is not None else None
        self._fallback = fallback or DeterministicFallbackRouter(
            dispatcher,
            policy=policy,
        )

    def route(self, question: str) -> ToolExecutionResult:
        if self._agent is None:
            return self._fallback.route(question)
        try:
            return self._agent.route(question)
        except (
            LMStudioError,
            LMStudioModelNotConfiguredError,
            ToolSelectionError,
            UnknownToolError,
            ToolArgumentsError,
            ToolExecutionError,
        ):
            return self._fallback.route(question)
