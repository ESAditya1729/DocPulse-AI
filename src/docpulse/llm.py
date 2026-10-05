"""IBM ICA Gateway LLM Client."""

from typing import Any

import httpx
from pydantic import BaseModel

from docpulse.config import Config


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict[str, Any] | None = None


class ICAGatewayClient:
    """Client for interacting with the IBM ICA Gateway."""

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.endpoint_url.rstrip("/")
        self.namespace = config.namespace.strip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

    def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models/assistants/agents under current namespace."""
        url = f"{self.base_url}/{self.namespace}/models"
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])

    def check_health(self) -> dict[str, Any]:
        """Perform a connection and health-check verification using OpenAPI namespace."""
        # 1. Try listing models under chosen namespace (e.g. /chat-models/models or friendly alias /chat-models)
        candidate_urls = [
            f"{self.base_url}/{self.namespace}/models",
            f"{self.base_url}/{self.namespace}",
        ]
        
        last_error_detail = ""
        for url in candidate_urls:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(url, headers=self.headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = data.get("data", []) if isinstance(data, dict) else []
                        return {
                            "status": "ok",
                            "endpoint": url,
                            "code": 200,
                            "models_count": len(models),
                        }
                    elif resp.status_code in (401, 403):
                        return {
                            "status": "error",
                            "message": "Authentication failed: Invalid or unauthorized API key.",
                            "code": resp.status_code,
                        }
                    else:
                        try:
                            body = resp.json()
                            last_error_detail = body.get("detail") or body.get("message") or resp.text
                        except (ValueError, AttributeError):
                            last_error_detail = resp.text[:200]
            except httpx.RequestError as e:
                return {"status": "error", "message": f"Network error connecting to {self.base_url}: {e}", "code": None}
            except Exception as e:  # noqa: BLE001 - check_health() must never raise; it reports status instead
                return {"status": "error", "message": str(e), "code": None}

        # 2. If listing models returned 400 (some gateways require POST chat/completions directly), test chat endpoint
        try:
            chat_url = f"{self.base_url}/{self.namespace}/chat/completions"
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    chat_url,
                    headers=self.headers,
                    json={
                        "model": self.config.model_id,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5,
                    },
                )
                if resp.status_code == 200:
                    return {
                        "status": "ok",
                        "endpoint": chat_url,
                        "code": 200,
                        "models_count": 1,
                    }
                elif resp.status_code in (401, 403):
                    return {
                        "status": "error",
                        "message": "Authentication failed: Invalid or unauthorized API key.",
                        "code": resp.status_code,
                    }
                else:
                    detail = ""
                    try:
                        err_json = resp.json()
                        detail = err_json.get("detail") or err_json.get("message") or resp.text[:200]
                    except (ValueError, AttributeError):
                        detail = resp.text[:200]
                    return {
                        "status": "warning",
                        "message": f"Server returned status {resp.status_code}: {detail or last_error_detail}",
                        "code": resp.status_code,
                    }
        except Exception as e:  # noqa: BLE001 - check_health() must never raise; it reports status instead
            return {"status": "error", "message": str(e), "code": None}

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

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                endpoint,
                headers=self.headers,
                json=payload,
            )
            if resp.is_error:
                error_detail = resp.text
                try:
                    err_json = resp.json()
                    error_detail = err_json.get("detail") or err_json.get("message") or err_json.get("error") or str(err_json)
                except (ValueError, AttributeError):
                    pass  # fall back to the raw response text already assigned above
                raise RuntimeError(f"HTTP {resp.status_code} from {endpoint}: {error_detail}")

            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            return LLMResponse(
                content=content,
                model=data.get("model", self.config.model_id),
                usage=data.get("usage"),
            )
