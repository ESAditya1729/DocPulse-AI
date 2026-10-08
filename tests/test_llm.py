"""Tests for gateway transport: truncation detection, retries, and error mapping."""

from unittest.mock import MagicMock

import httpx
import pytest

from docpulse.config import Config
from docpulse.errors import ConfigError, LLMError
from docpulse.llm import ICAGatewayClient, close_shared_client

CHAT_URL = "https://gateway.example.com/v1/chat-models/chat/completions"
API_KEY = "super-secret-key-value"


def build_client(**overrides) -> ICAGatewayClient:
    settings = {
        "endpoint_url": "https://gateway.example.com/v1",
        "api_key": API_KEY,
        "namespace": "chat-models",
        "model_id": "test-model",
        "retry_backoff": 0.0,
    }
    settings.update(overrides)
    return ICAGatewayClient(Config(**settings))


def make_response(
    status_code: int = 200,
    content: str = "hello",
    finish_reason: str | None = "stop",
    usage: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    body = {"model": "test-model", "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(
        status_code=status_code,
        json=body,
        headers=headers or {},
        request=httpx.Request("POST", CHAT_URL),
    )


@pytest.fixture
def http(monkeypatch):
    """Replace the shared HTTP client with a mock and disable real sleeping."""
    fake = MagicMock()
    monkeypatch.setattr("docpulse.llm.get_shared_client", lambda: fake)
    monkeypatch.setattr("docpulse.llm.time.sleep", lambda _seconds: None)
    yield fake
    close_shared_client()


# --- Success path -----------------------------------------------------------


def test_chat_complete_captures_finish_reason_and_usage(http):
    http.post.return_value = make_response(finish_reason="stop", usage={"total_tokens": 42})

    response = build_client().chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "hello"
    assert response.finish_reason == "stop"
    assert response.truncated is False
    assert response.usage == {"total_tokens": 42}


def test_length_finish_reason_is_reported_as_truncated(http):
    """The gateway still returns 200 here, so truncation must be detected explicitly."""
    http.post.return_value = make_response(content='{"concepts": [{"name": "half', finish_reason="length")

    response = build_client().chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert response.truncated is True
    assert response.finish_reason == "length"


def test_chat_complete_passes_model_and_generation_settings(http):
    http.post.return_value = make_response()

    build_client(temperature=0.7, max_tokens=512).chat_complete(messages=[{"role": "user", "content": "hi"}])

    payload = http.post.call_args.kwargs["json"]
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 512


# --- Retry behaviour --------------------------------------------------------


def test_retries_throttling_then_succeeds(http):
    http.post.side_effect = [
        make_response(status_code=429, headers={"Retry-After": "0"}),
        make_response(content="second try"),
    ]

    response = build_client(max_retries=3).chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "second try"
    assert http.post.call_count == 2


def test_gives_up_after_max_retries(http):
    http.post.return_value = make_response(status_code=503)

    with pytest.raises(LLMError) as excinfo:
        build_client(max_retries=2).chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert "503" in excinfo.value.message
    assert http.post.call_count == 3, "initial attempt plus two retries"


def test_zero_retries_means_a_single_attempt(http):
    http.post.return_value = make_response(status_code=503)

    with pytest.raises(LLMError):
        build_client(max_retries=0).chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert http.post.call_count == 1


def test_client_errors_are_not_retried(http):
    http.post.return_value = make_response(status_code=400)

    with pytest.raises(LLMError):
        build_client(max_retries=3).chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert http.post.call_count == 1, "a malformed request will fail identically every time"


def test_retry_callback_reports_attempt_count_and_delay(http):
    seen: list[tuple[int, int, float]] = []
    http.post.side_effect = [make_response(status_code=500), make_response()]

    client = build_client(max_retries=2, retry_backoff=1.5)
    client.on_retry = lambda attempt, attempts, delay, _err: seen.append((attempt, attempts, delay))
    client.chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert len(seen) == 1
    attempt, attempts, delay = seen[0]
    assert (attempt, attempts) == (1, 3), "first failure of a run that allows three attempts"
    assert 1.5 <= delay <= 1.5 * 1.25, "backoff is exponential with up to 25% jitter"


def test_retry_after_header_overrides_backoff(http):
    slept: list[float] = []
    http.post.side_effect = [
        make_response(status_code=429, headers={"Retry-After": "7"}),
        make_response(),
    ]

    client = build_client(max_retries=2, retry_backoff=10.0)
    client.on_retry = lambda _a, _b, _d, _e: slept.append(7.0)
    client.chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert slept == [7.0], "server-directed delay wins over local backoff"


# --- Error mapping ----------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_are_reported_as_config_errors(http, status):
    http.post.return_value = make_response(status_code=status)

    with pytest.raises(ConfigError) as excinfo:
        build_client().chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert "docpulse init" in (excinfo.value.hint or "")


def test_404_is_reported_as_a_config_error_naming_the_namespace(http):
    http.post.return_value = make_response(status_code=404)

    with pytest.raises(ConfigError) as excinfo:
        build_client().chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert "404" in excinfo.value.message
    assert "chat-models" in excinfo.value.message


def test_timeouts_become_gateway_errors_naming_the_limit(http):
    http.post.side_effect = httpx.TimeoutException("read timed out")

    with pytest.raises(LLMError) as excinfo:
        build_client(request_timeout=45.0, max_retries=0).chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert "45" in excinfo.value.message
    assert "DOCPULSE_TIMEOUT" in (excinfo.value.hint or "")


def test_connection_errors_become_gateway_errors_pointing_at_init(http):
    http.post.side_effect = httpx.ConnectError("name resolution failed")

    with pytest.raises(LLMError) as excinfo:
        build_client(max_retries=0).chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert "docpulse init" in (excinfo.value.hint or "")


def test_api_key_is_redacted_from_error_bodies(http):
    leaky = httpx.Response(
        status_code=400,
        json={"detail": f"invalid key {API_KEY} supplied"},
        request=httpx.Request("POST", CHAT_URL),
    )
    http.post.return_value = leaky

    with pytest.raises(LLMError) as excinfo:
        build_client(max_retries=0).chat_complete(messages=[{"role": "user", "content": "hi"}])

    assert API_KEY not in excinfo.value.message
    assert "***" in excinfo.value.message


def test_response_without_choices_raises_instead_of_keyerror(http):
    empty = httpx.Response(status_code=200, json={"model": "m"}, request=httpx.Request("POST", CHAT_URL))
    http.post.return_value = empty

    with pytest.raises(LLMError, match="no choices"):
        build_client().chat_complete(messages=[{"role": "user", "content": "hi"}])


def test_non_json_gateway_response_raises_llm_error(http):
    http.post.return_value = httpx.Response(status_code=200, text="<html>502 Bad Gateway</html>", request=httpx.Request("POST", CHAT_URL))

    with pytest.raises(LLMError, match="non-JSON"):
        build_client().chat_complete(messages=[{"role": "user", "content": "hi"}])


# --- Discovery / health -----------------------------------------------------


def test_list_models_raises_gateway_errors(http):
    http.get.return_value = httpx.Response(status_code=401, json={"detail": "nope"}, request=httpx.Request("GET", CHAT_URL))

    with pytest.raises(ConfigError):
        build_client().list_models()


def test_check_health_reports_rather_than_raises(http):
    """check_health is used during `init`, where an exception would abort setup."""
    http.get.side_effect = httpx.ConnectError("refused")

    status = build_client().check_health()

    assert status["status"] == "error"
    assert "refused" in status["message"]