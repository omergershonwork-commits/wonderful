"""Offline tests for the LM Studio client boundary."""

import httpx
import pytest

from src.config import Settings
from src.llm_client import (
    LMStudioClient,
    LMStudioDisabledError,
    LMStudioHealthState,
    LMStudioHTTPError,
    LMStudioInvalidResponseError,
    LMStudioTimeoutError,
)


def _transport(handler):
    return httpx.MockTransport(handler)


def test_list_models_validates_response_and_auth_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://localhost:1234/v1/models")
        assert request.headers["Authorization"] == "Bearer local-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": " qwen3-8b ", "object": "model", "owned_by": "local"},
                    {"id": "qwen3-8b"},
                    {"id": "embedding-model"},
                ]
            },
        )

    client = LMStudioClient(
        base_url=" http://localhost:1234/v1/ ",
        api_key="local-key",
        model="qwen3-8b",
        transport=_transport(handler),
    )
    assert client.base_url == "http://localhost:1234/v1/"
    assert [model.id for model in client.list_models()] == [
        "qwen3-8b",
        "embedding-model",
    ]


def test_whitespace_only_model_id_is_rejected() -> None:
    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        transport=_transport(
            lambda request: httpx.Response(200, json={"data": [{"id": "   "}]})
        ),
    )
    with pytest.raises(LMStudioInvalidResponseError):
        client.list_models()
    assert client.health_check().state is LMStudioHealthState.INVALID_RESPONSE


def test_health_check_reports_ready_when_configured_model_exists() -> None:
    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        model="qwen3-8b",
        transport=_transport(
            lambda request: httpx.Response(200, json={"data": [{"id": "qwen3-8b"}]})
        ),
    )
    status = client.health_check()
    assert status.state is LMStudioHealthState.READY
    assert status.running is True
    assert status.ready is True
    assert status.configured_model_available is True


def test_health_check_detects_running_server_with_missing_model() -> None:
    status = LMStudioClient(
        base_url="http://localhost:1234/v1",
        model="qwen3-8b",
        transport=_transport(
            lambda request: httpx.Response(200, json={"data": [{"id": "other-model"}]})
        ),
    ).health_check()
    assert status.state is LMStudioHealthState.RUNNING
    assert status.running is True
    assert status.ready is False


def test_disabled_mode_never_touches_the_network() -> None:
    def fail_if_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("disabled mode must not perform HTTP requests")

    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        enabled=False,
        transport=_transport(fail_if_called),
    )
    assert client.health_check().state is LMStudioHealthState.DISABLED
    with pytest.raises(LMStudioDisabledError):
        client.list_models()


def test_timeout_is_typed_and_health_check_is_non_throwing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow server", request=request)

    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        transport=_transport(handler),
    )
    with pytest.raises(LMStudioTimeoutError):
        client.list_models()
    assert client.health_check().state is LMStudioHealthState.TIMEOUT


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not-json"),
        httpx.Response(200, json={"data": {}}),
        httpx.Response(200, json={"data": [{"object": "model"}]}),
        httpx.Response(200, json={"data": [{"id": "   "}]}),
    ],
)
def test_invalid_responses_are_rejected(response: httpx.Response) -> None:
    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        transport=_transport(lambda request: response),
    )
    with pytest.raises(LMStudioInvalidResponseError):
        client.list_models()


def test_http_failure_is_typed_and_reported_unavailable() -> None:
    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        transport=_transport(lambda request: httpx.Response(503)),
    )
    with pytest.raises(LMStudioHTTPError):
        client.list_models()
    assert client.health_check().state is LMStudioHealthState.UNAVAILABLE


def test_from_settings_uses_runtime_configuration() -> None:
    settings = Settings(
        llm_base_url="http://127.0.0.1:9999/v1",
        llm_api_key="configured-key",
        llm_model="configured-model",
        use_llm=False,
        http_timeout_seconds=7,
    )
    client = LMStudioClient.from_settings(settings)
    assert client.base_url == "http://127.0.0.1:9999/v1/"
    assert client.configured_model == "configured-model"
    assert client.enabled is False
