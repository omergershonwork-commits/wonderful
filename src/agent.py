"""Validated LM Studio tool calling for the five approved airport tools."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from pydantic import BaseModel, ValidationError

from src.agent_contracts import (
    AgentError,
    LMStudioModelNotConfiguredError,
    RouteSource,
    RoutingPolicy,
    SelectedToolCall,
    ToolArgumentsError,
    ToolExecutionError,
    ToolExecutionResult,
    ToolSelectionError,
    UnknownToolError,
)
from src.exceptions import DataNotFoundError
from src.llm_client import (
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioHTTPError,
    LMStudioInvalidResponseError,
    LMStudioTimeoutError,
)
from src.models import (
    CalculateLongHaulShareInput,
    CompareAirportsInput,
    EstimateUnmetCapacityInput,
    GetAirportProfileInput,
    RankAirportsInput,
)
from src.question_constraints import (
    QuestionConstraints,
    apply_numeric_policy_defaults,
    parse_question_constraints,
    validate_question_semantics,
)
from src.tools import AirportAnalyticsService

__all__ = [
    "AgentError", "AgentToolCaller", "LMStudioChatClient",
    "LMStudioModelNotConfiguredError", "QuestionConstraints", "RouteSource",
    "RoutingPolicy", "SelectedToolCall", "ToolArgumentsError", "ToolDispatcher",
    "ToolExecutionError", "ToolExecutionResult", "ToolSelectionError",
    "UnknownToolError", "approved_tool_schemas", "parse_question_constraints",
]


class LMStudioChatClient(LMStudioClient):
    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self._require_enabled()
        if not self.configured_model:
            raise LMStudioModelNotConfiguredError("An exact LM Studio model ID must be configured")
        if not messages:
            raise ValueError("messages must not be empty")
        payload: dict[str, Any] = {
            "model": self.configured_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        try:
            response = self._client.post("chat/completions", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LMStudioTimeoutError("LM Studio timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise LMStudioHTTPError(f"LM Studio returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise LMStudioConnectionError("Could not connect to LM Studio") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise LMStudioInvalidResponseError("LM Studio returned non-JSON chat output") from exc
        if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or not body["choices"]:
            raise LMStudioInvalidResponseError("Chat output must contain at least one choice")
        return body


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    service_method: str

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


APPROVED_TOOL_SPECS = (
    ToolSpec("rank_airports", "Rank supported airports.", RankAirportsInput, "rank_airports"),
    ToolSpec("compare_airports", "Compare supported airports.", CompareAirportsInput, "compare_airports"),
    ToolSpec("calculate_long_haul_share", "Calculate long-haul shares.", CalculateLongHaulShareInput, "calculate_long_haul_share"),
    ToolSpec("estimate_unmet_capacity", "Estimate unmet-capacity proxy.", EstimateUnmetCapacityInput, "estimate_unmet_capacity"),
    ToolSpec("get_airport_profile", "Return one airport profile.", GetAirportProfileInput, "get_airport_profile"),
)

TOOL_SYSTEM_PROMPT = """Select exactly one approved tool. Never answer in prose.
Preserve all airports, metrics, exclusions, filters, and numeric overrides.
Python applies runtime defaults and performs calculations."""


def approved_tool_schemas() -> list[dict[str, Any]]:
    return [item.openai_schema() for item in APPROVED_TOOL_SPECS]


class ToolDispatcher:
    def __init__(self, service: AirportAnalyticsService | None = None) -> None:
        self._service = service or AirportAnalyticsService()
        self._specs = {item.name: item for item in APPROVED_TOOL_SPECS}

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def validate_arguments(self, tool_name: str, arguments: dict[str, Any]) -> BaseModel:
        spec = self._specs.get(tool_name)
        if spec is None:
            raise UnknownToolError(f"Unknown or unapproved tool: {tool_name}")
        try:
            return spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentsError(f"Arguments for {tool_name} failed validation") from exc

    def execute(
        self,
        *,
        question: str,
        tool_name: str,
        arguments: dict[str, Any],
        route_source: RouteSource = RouteSource.LLM,
    ) -> ToolExecutionResult:
        request = self.validate_arguments(tool_name, arguments)
        spec = self._specs[tool_name]
        executor: Callable[[BaseModel], Any] = getattr(self._service, spec.service_method)
        try:
            output = executor(request)
        except (ValidationError, ValueError, KeyError, DataNotFoundError) as exc:
            raise ToolExecutionError(f"Tool {tool_name} failed: {exc}") from exc
        return ToolExecutionResult(
            question=question,
            tool_name=tool_name,
            arguments=request.model_dump(mode="json"),
            output=output,
            route_source=route_source,
        )


class AgentToolCaller:
    def __init__(
        self,
        client: LMStudioChatClient,
        *,
        dispatcher: ToolDispatcher | None = None,
        policy: RoutingPolicy | None = None,
    ) -> None:
        self._client = client
        self._dispatcher = dispatcher or ToolDispatcher()
        self._policy = policy or RoutingPolicy()

    @property
    def dispatcher(self) -> ToolDispatcher:
        return self._dispatcher

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    def route(self, question: str) -> ToolExecutionResult:
        normalized = question.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        response = self._client.create_chat_completion(
            messages=[
                {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": normalized},
            ],
            tools=approved_tool_schemas(),
            tool_choice="required",
            temperature=0.0,
        )
        selected = self._parse_selected_tool(response)
        if selected.name not in self._dispatcher.tool_names:
            raise UnknownToolError(f"Unknown or unapproved tool: {selected.name}")
        constraints = parse_question_constraints(normalized)
        arguments = apply_numeric_policy_defaults(selected, constraints, self._policy)
        request = self._dispatcher.validate_arguments(selected.name, arguments)
        validate_question_semantics(constraints, selected.name, request, self._policy)
        return self._dispatcher.execute(
            question=normalized,
            tool_name=selected.name,
            arguments=request.model_dump(mode="json"),
            route_source=RouteSource.LLM,
        )

    @staticmethod
    def _parse_selected_tool(response: dict[str, Any]) -> SelectedToolCall:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ToolSelectionError("Chat response has no assistant message") from exc
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(calls, list) or len(calls) != 1:
            raise ToolSelectionError("The model must return exactly one structured tool call")
        function = calls[0].get("function") if isinstance(calls[0], dict) else None
        if not isinstance(function, dict):
            raise ToolSelectionError("Tool call must contain a function object")
        name, raw = function.get("name"), function.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise ToolSelectionError("Tool call function name is missing")
        if isinstance(raw, str):
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolArgumentsError("Tool arguments are not valid JSON") from exc
        else:
            arguments = raw
        if not isinstance(arguments, dict):
            raise ToolArgumentsError("Tool arguments must be a JSON object")
        return SelectedToolCall(name=name.strip(), arguments=arguments)
