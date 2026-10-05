"""Content-addressed local cache for LLM responses.

Keys off exactly what would be sent to the gateway (endpoint, namespace,
model, system prompt, messages), so switching any of those naturally
invalidates the cache - no explicit expiry needed.
"""

import hashlib
import json
import os
from pathlib import Path

from docpulse.config import Config
from docpulse.llm import ICAGatewayClient

DEFAULT_CACHE_DIR = Path.home() / ".docpulse" / "cache"


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
            "system_prompt": system_prompt,
            "messages": messages,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load(key: str) -> str | None:
    path = get_cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["content"]
    except (OSError, ValueError, KeyError):
        return None


def _store(key: str, content: str) -> None:
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"content": content}, f)


def get_cached_or_generate(
    client: ICAGatewayClient,
    config: Config,
    *,
    system_prompt: str | None,
    messages: list[dict],
    use_cache: bool = True,
) -> tuple[str, bool]:
    """Return (content, was_cached). Bypasses lookup/store entirely when use_cache is False."""
    if not use_cache:
        return client.chat_complete(system_prompt=system_prompt, messages=messages).content, False

    key = _cache_key(config, system_prompt, messages)
    cached = _load(key)
    if cached is not None:
        return cached, True

    content = client.chat_complete(system_prompt=system_prompt, messages=messages).content
    _store(key, content)
    return content, False
