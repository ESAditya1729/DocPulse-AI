"""IBM ICA Gateway LLM Client."""

import random
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel

from docpulse.config import SETUP_HINT, Config
from docpulse.errors import ConfigError, DocPulseError, LLMError

# Statuses worth retrying: rate limiting plus transient upstream failures.
# Any other 4xx is a bad request and retrying it just wastes the user's time.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Never sleep longer than this between attempts, even if Retry-After asks for it.
MAX_RETRY_DELAY = 30.0


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """True when the model stopped because it hit the token limit.

        The gateway still returns HTTP 200 for these, and the content arrives
        mid-sentence - which for our JSON commands means a payload that parses
        into nothing. Callers must check this rather than reporting "no
        concepts found".
        """
        return self.finish_reason == "length"


# One pooled client per process. Building a fresh httpx.Client per request (as
# an earlier version did) threw away TCP/TLS reuse between a document's
# several calls, which is the dominant cost for short gateway round-trips.
_shared_client: httpx.Client | None = None


def get_shared_client() -> httpx.Client:
    """Return the process-wide HTTP client, creating it on first use."""
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.Client(limits=httpx.Limits(max_connections=16, max_keepalive_connections=8))
    return _shared_client


def close_shared_client() -> None:
    """Close the shared client. Safe to call when none was ever created."""
    global _shared_client
    if _shared_client is not None:
        _shared_client.close()
        _shared_client = None


def _parse_retry_after(resp: httpx.Response) -> float | None:
    """Read a numeric Retry-After header, ignoring HTTP-date form."""
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


class ICAGatewayClient:
    """Client for interacting with the IBM ICA Gateway."""

    def __init__(self, config: Config, on_retry: Callable[[int, int, float, DocPulseError], None] | None = None):
        self.config = config
        self.base_url = config.endpoint_url.rstrip("/")
        self.namespace = config.namespace.strip("/")
        self.on_retry = on_retry
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

    # --- internals ---------------------------------------------------------

    def _redact(self, text: str) -> str:
        """Strip the API key out of text destined for an error message.

        Gateways sometimes echo the Authorization header back in an error body;
        without this the key would land in a terminal scrollback or a log.
        """
        key = self.config.api_key
        if key and key in text:
            return text.replace(key, "***")
        return text

    def _error_detail(self, resp: httpx.Response) -> str:
        """Pull the most useful error text out of a failed response."""
        try:
            body = resp.json()
        except ValueError:
            return self._redact(resp.text[:200])
        if isinstance(body, dict):
            for field in ("detail", "message", "error"):
                if body.get(field):
                    return self._redact(str(body[field])[:200])
        return self._redact(str(body)[:200])

    def _http_error(self, resp: httpx.Response) -> DocPulseError:
        """Map an HTTP error response to the most specific error type."""
        url = str(resp.url)
        if resp.status_code in (401, 403):
            return ConfigError(
                f"Gateway rejected the API key (HTTP {resp.status_code}) at {url}.",
                hint="Re-run 'docpulse init' to re-enter the key, or update DOCPULSE_API_KEY.",
            )
        if resp.status_code == 404:
            return ConfigError(
                f"Gateway returned 404 for {url} - the endpoint URL or namespace '{self.namespace}' is wrong.",
                hint=SETUP_HINT,
            )
        return LLMError(f"Gateway returned HTTP {resp.status_code} for {url}: {self._error_detail(resp)}")

    def _request_timeout(self) -> httpx.Timeout:
        total = self.config.request_timeout
        return httpx.Timeout(total, connect=min(10.0, total))

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped so retries stay bounded."""
        base = self.config.retry_backoff * (2 ** (attempt - 1))
        return min(base * (1.0 + random.random() * 0.25), MAX_RETRY_DELAY)  # noqa: S311 - retry jitter, not security

    def _post_with_retries(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        """POST with retry on throttling/transient failures; raise on give-up."""
        attempts = self.config.max_retries + 1
        client = get_shared_client()

        for attempt in range(1, attempts + 1):
            retry_after: float | None = None
            try:
                resp = client.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=self._request_timeout(),
                )
            except httpx.TimeoutException as exc:
                error: DocPulseError = LLMError(
                    f"Timed out after {self.config.request_timeout:g}s waiting for {url} "
                    f"(attempt {attempt}/{attempts}): {self._redact(str(exc))}",
                    hint="Raise the limit with DOCPULSE_TIMEOUT=300, or check whether the gateway is overloaded.",
                )
            except httpx.RequestError as exc:
                error = LLMError(
                    f"Could not reach {self.base_url} (attempt {attempt}/{attempts}): "
                    f"{self._redact(f'{type(exc).__name__}: {exc}')}",
                    hint="Verify the endpoint URL with 'docpulse init' and check your network or proxy settings.",
                )
            else:
                if not resp.is_error:
                    return resp
                error = self._http_error(resp)
                if resp.status_code not in RETRYABLE_STATUS:
                    raise error
                retry_after = _parse_retry_after(resp)

            if attempt == attempts:
                raise error

            delay = retry_after if retry_after is not None else self._backoff_delay(attempt)
            delay = min(delay, MAX_RETRY_DELAY)
            if self.on_retry is not None:
                self.on_retry(attempt, attempts, delay, error)
            time.sleep(delay)

        # Unreachable: the loop either returns a response or raises.
        raise LLMError(f"Gateway request to {url} failed after {attempts} attempts: {error.message}")

    # --- public API --------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models/assistants/agents under current namespace."""
        url = f"{self.base_url}/{self.namespace}/models"
        client = get_shared_client()
        try:
            resp = client.get(url, headers=self.headers, timeout=httpx.Timeout(10.0))
        except httpx.TimeoutException as exc:
            raise LLMError(f"Timed out after 10s listing models at {url}: {self._redact(str(exc))}") from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"Could not reach {self.base_url} while listing models: {self._redact(f'{type(exc).__name__}: {exc}')}",
                hint="Verify the endpoint URL with 'docpulse init'.",
            ) from exc
        if resp.is_error:
            raise self._http_error(resp)
        data = resp.json()
        if not isinstance(data, dict):
            return []
        models = data.get("data", [])
        return [m for m in models if isinstance(m, dict)] if isinstance(models, list) else []

    def check_health(self) -> dict[str, Any]:
        """Perform a connection and health-check verification using OpenAPI namespace."""
        # 1. Try listing models under chosen namespace (e.g. /chat-models/models or friendly alias /chat-models)
        candidate_urls = [
            f"{self.base_url}/{self.namespace}/models",
            f"{self.base_url}/{self.namespace}",
        ]

        last_error_detail = ""
        client = get_shared_client()
        for url in candidate_urls:
            try:
                resp = client.get(url, headers=self.headers, timeout=httpx.Timeout(10.0))
            except httpx.TimeoutException as exc:
                return {
                    "status": "error",
                    "message": f"Timed out after 10s connecting to {self.base_url}: {self._redact(str(exc))}",
                    "code": None,
                }
            except httpx.RequestError as exc:
                return {
                    "status": "error",
                    "message": f"Network error connecting to {self.base_url}: "
                    f"{self._redact(f'{type(exc).__name__}: {exc}')}",
                    "code": None,
                }
            except Exception as e:  # noqa: BLE001 - check_health() must never raise; it reports status instead
                return {"status": "error", "message": self._redact(str(e)), "code": None}

            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", []) if isinstance(data, dict) else []
                return {
                    "status": "ok",
                    "endpoint": url,
                    "code": 200,
                    "models_count": len(models) if isinstance(models, list) else 0,
                }
            if resp.status_code in (401, 403):
                return {
                    "status": "error",
                    "message": "Authentication failed: Invalid or unauthorized API key.",
                    "code": resp.status_code,
                }
            try:
                body = resp.json()
                last_error_detail = body.get("detail") or body.get("message") or self._redact(resp.text[:200])
            except (ValueError, AttributeError):
                last_error_detail = self._redact(resp.text[:200])

        # 2. If listing models returned 400 (some gateways require POST chat/completions directly), test chat endpoint
        chat_url = f"{self.base_url}/{self.namespace}/chat/completions"
        try:
            resp = client.post(
                chat_url,
                headers=self.headers,
                json={
                    "model": self.config.model_id,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
                timeout=httpx.Timeout(30.0),
            )
        except Exception as e:  # noqa: BLE001 - check_health() must never raise; it reports status instead
            return {"status": "error", "message": self._redact(str(e)), "code": None}

        if resp.status_code == 200:
            return {"status": "ok", "endpoint": chat_url, "code": 200, "models_count": 1}
        if resp.status_code in (401, 403):
            return {
                "status": "error",
                "message": "Authentication failed: Invalid or unauthorized API key.",
                "code": resp.status_code,
            }
        return {
            "status": "warning",
            "message": f"Server returned status {resp.status_code}: {last_error_detail or self._error_detail(resp)}",
            "code": resp.status_code,
        }

    def chat_complete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send chat messages and get complete generation response."""
        req_messages = []
        if system_prompt:
            req_messages.append({"role": "system", "content": system_prompt})
        req_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": req_messages,
        }

        # Only send temperature if it differs from 0.2 (LiteLLM proxy rejects an explicit
        # default temperature on some reasoning / strict models)
        temp_val = temperature if temperature is not None else self.config.temperature
        if temp_val is not None and temp_val != 0.2:
            payload["temperature"] = temp_val

        tokens_val = max_tokens if max_tokens is not None else self.config.max_tokens
        if tokens_val is not None:
            payload["max_tokens"] = tokens_val

        endpoint = f"{self.base_url}/{self.namespace}/chat/completions"
        resp = self._post_with_retries(endpoint, payload)

        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"Gateway returned a non-JSON response from {endpoint}: {self._redact(resp.text[:200])}") from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LLMError(f"Gateway response from {endpoint} contained no choices: {self._redact(str(data)[:200])}")

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        if content is None:
            raise LLMError(f"Gateway response from {endpoint} contained no message content.")

        usage = data.get("usage") if isinstance(data, dict) else None
        return LLMResponse(
            content=str(content),
            model=data.get("model") or self.config.model_id,
            usage=usage if isinstance(usage, dict) else None,
            finish_reason=first.get("finish_reason"),
        )