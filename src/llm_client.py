"""Small, typed client for the LM Studio OpenAI-compatible local API."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import Settings


class LMStudioError(RuntimeError):
    """Base error raised by the LM Studio boundary."""


class LMStudioDisabledError(LMStudioError):
    """Raised when an operation requires an explicitly disabled client."""


class LMStudioTimeoutError(LMStudioError):
    """Raised when LM Studio does not respond within the configured timeout."""


class LMStudioConnectionError(LMStudioError):
    """Raised when the local LM Studio server cannot be reached."""


class LMStudioHTTPError(LMStudioError):
    """Raised when LM Studio returns a non-success HTTP response."""


class LMStudioInvalidResponseError(LMStudioError):
    """Raised when LM Studio returns JSON that violates the expected contract."""


class LMStudioHealthState(StrEnum):
    """Stable application-facing LM Studio availability states."""

    DISABLED = "disabled"
    READY = "ready"
    RUNNING = "running"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"


class LMStudioModel(BaseModel):
    """One nonblank model returned by LM Studio's models endpoint."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    object: str | None = None
    owned_by: str | None = None


class LMStudioHealth(BaseModel):
    """Non-throwing health snapshot used by the application and UI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: LMStudioHealthState
    running: bool
    ready: bool
    enabled: bool
    base_url: str
    configured_model: str | None = None
    configured_model_available: bool = False
    models: tuple[LMStudioModel, ...] = ()
    message: str
    error_code: str | None = None


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip()
    if not normalized:
        raise ValueError("LM Studio base URL cannot be empty")
    return normalized.rstrip("/") + "/"


class LMStudioClient:
    """Synchronous client for LM Studio health and model discovery."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "lm-studio",
        model: str | None = None,
        timeout_seconds: float = 20,
        enabled: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._base_url = _normalize_base_url(base_url)
        self._model = model.strip() if model and model.strip() else None
        self._enabled = enabled
        headers = {"Accept": "application/json"}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers=headers,
            transport=transport,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> "LMStudioClient":
        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.http_timeout_seconds,
            enabled=settings.use_llm,
            transport=transport,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def configured_model(self) -> str | None:
        return self._model

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LMStudioClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise LMStudioDisabledError("LM Studio integration is disabled by configuration")

    def _get_json(self, path: str) -> Any:
        self._require_enabled()
        try:
            response = self._client.get(path)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LMStudioTimeoutError(
                "LM Studio did not respond before the configured timeout"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LMStudioHTTPError(
                f"LM Studio returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise LMStudioConnectionError("Could not connect to LM Studio") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise LMStudioInvalidResponseError(
                "LM Studio returned a non-JSON response"
            ) from exc

    def list_models(self) -> tuple[LMStudioModel, ...]:
        """Return validated unique nonblank model IDs in reported order."""

        payload = self._get_json("models")
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise LMStudioInvalidResponseError(
                "LM Studio models response must contain a data list"
            )

        models: list[LMStudioModel] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(payload["data"]):
            try:
                model = LMStudioModel.model_validate(item)
            except (ValidationError, TypeError) as exc:
                raise LMStudioInvalidResponseError(
                    f"LM Studio model at index {index} is invalid"
                ) from exc
            if model.id not in seen_ids:
                seen_ids.add(model.id)
                models.append(model)
        return tuple(models)

    def health_check(self) -> LMStudioHealth:
        if not self._enabled:
            return LMStudioHealth(
                state=LMStudioHealthState.DISABLED,
                running=False,
                ready=False,
                enabled=False,
                base_url=self._base_url,
                configured_model=self._model,
                message="LM Studio integration is disabled.",
            )

        try:
            models = self.list_models()
        except LMStudioTimeoutError as exc:
            return self._error_health(
                LMStudioHealthState.TIMEOUT, "LM_STUDIO_TIMEOUT", str(exc)
            )
        except LMStudioInvalidResponseError as exc:
            return self._error_health(
                LMStudioHealthState.INVALID_RESPONSE,
                "LM_STUDIO_INVALID_RESPONSE",
                str(exc),
            )
        except (LMStudioConnectionError, LMStudioHTTPError) as exc:
            return self._error_health(
                LMStudioHealthState.UNAVAILABLE,
                "LM_STUDIO_UNAVAILABLE",
                str(exc),
            )

        model_ids = {model.id for model in models}
        configured_available = self._model is not None and self._model in model_ids
        if configured_available:
            state = LMStudioHealthState.READY
            message = f"LM Studio is running and model '{self._model}' is available."
        elif self._model is None:
            state = LMStudioHealthState.RUNNING
            message = "LM Studio is running, but no model ID is configured."
        elif not models:
            state = LMStudioHealthState.RUNNING
            message = "LM Studio is running, but it reported no loaded models."
        else:
            state = LMStudioHealthState.RUNNING
            message = (
                f"LM Studio is running, but configured model '{self._model}' "
                "was not reported."
            )

        return LMStudioHealth(
            state=state,
            running=True,
            ready=configured_available,
            enabled=True,
            base_url=self._base_url,
            configured_model=self._model,
            configured_model_available=configured_available,
            models=models,
            message=message,
        )

    def _error_health(
        self,
        state: LMStudioHealthState,
        error_code: str,
        message: str,
    ) -> LMStudioHealth:
        return LMStudioHealth(
            state=state,
            running=False,
            ready=False,
            enabled=True,
            base_url=self._base_url,
            configured_model=self._model,
            message=message,
            error_code=error_code,
        )
