"""Shared agent contracts used by LLM and deterministic routing."""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentError(RuntimeError):
    pass


class LMStudioModelNotConfiguredError(AgentError):
    pass


class ToolSelectionError(AgentError):
    pass


class UnknownToolError(AgentError):
    pass


class ToolArgumentsError(AgentError):
    pass


class ToolExecutionError(AgentError):
    pass


class RouteSource(StrEnum):
    LLM = "llm"
    FALLBACK = "fallback"


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)
    question: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    output: Any
    route_source: RouteSource = RouteSource.LLM


class SelectedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class RoutingPolicy(BaseModel):
    """Runtime defaults shared by model routing, fallback, context, and UI."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ranking_limit: int = Field(default=5, ge=1, le=10)
    long_haul_threshold_miles: int = Field(default=3000, gt=0)
    target_load_factor: float = Field(default=0.82, gt=0, le=1)
