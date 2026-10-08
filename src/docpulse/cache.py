"""Content-addressed local cache for LLM responses.

Keys off exactly what would be sent to the gateway (endpoint, namespace,
model, system prompt, messages, and the sampling settings), so switching any
of those naturally invalidates the cache - no explicit expiry needed.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, NamedTuple

from docpulse.config import Config
from docpulse.llm import ICAGatewayClient, LLMResponse

DEFAULT_CACHE_DIR = Path.home() / ".docpulse" / "cache"


class LLMOutcome(NamedTuple):
    """A generated (or cached) response plus the metadata callers need to warn."""

    content: str
    was_cached: bool
    truncated: bool = False
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


def get_cache_dir() -> Path:
    """Return the cache directory, allowing override via env var."""
    custom_path = os.getenv("DOCPULSE_CACHE_DIR")
    if custom_path:
        return Path(custom_path)
    return DEFAULT_CACHE_DIR


def _cache_key(config: Config, system_prompt: str | None, messages: list[dict]) -> str:
    payload = json.dumps(
        {
            "endpoint_url": config.endpoint_url,
            "namespace": config.namespace,
            "model_id": config.model_id,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "system_prompt": system_prompt,
            "messages": messages,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load(key: str) -> LLMOutcome | None:
    """Read a cached outcome, tolerating legacy entries that stored only content."""
    path = get_cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None

    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, str):
        return None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    finish_reason = payload.get("finish_reason") if isinstance(payload, dict) else None
    return LLMOutcome(
        content=content,
        was_cached=True,
        truncated=finish_reason == "length",
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        usage=usage if isinstance(usage, dict) else None,
    )


def _store(key: str, response: LLMResponse) -> None:
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "content": response.content,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            },
            f,
        )


def _to_outcome(response: LLMResponse, *, was_cached: bool) -> LLMOutcome:
    return LLMOutcome(
        content=response.content,
        was_cached=was_cached,
        truncated=response.truncated,
        finish_reason=response.finish_reason,
        usage=response.usage,
    )


def get_cached_or_generate(
    client: ICAGatewayClient,
    config: Config,
    *,
    system_prompt: str | None,
    messages: list[dict],
    use_cache: bool = True,
) -> LLMOutcome:
    """Return an LLMOutcome. Bypasses lookup/store entirely when use_cache is False.

    Truncated responses are deliberately never written to the cache: caching
    half an answer would make `--max-tokens 4096` a no-op on the next run,
    because the incomplete response would be served straight back.
    """
    if not use_cache:
        return _to_outcome(client.chat_complete(system_prompt=system_prompt, messages=messages), was_cached=False)

    key = _cache_key(config, system_prompt, messages)
    cached = _load(key)
    if cached is not None:
        return cached

    response = client.chat_complete(system_prompt=system_prompt, messages=messages)
    if not response.truncated:
        _store(key, response)
    return _to_outcome(response, was_cached=False)