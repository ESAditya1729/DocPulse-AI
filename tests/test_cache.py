"""Tests for the local content-addressed response cache."""

import json
from unittest.mock import MagicMock

from docpulse.cache import _cache_key, get_cache_dir, get_cached_or_generate
from docpulse.config import Config
from docpulse.llm import LLMResponse


def make_client(content: str, finish_reason: str | None = None) -> MagicMock:
    client = MagicMock()
    client.chat_complete.return_value = LLMResponse(content=content, model="test-model", finish_reason=finish_reason)
    return client


def test_cache_dir_respects_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "my-cache"
    monkeypatch.setenv("DOCPULSE_CACHE_DIR", str(custom))
    assert get_cache_dir() == custom


def test_second_call_is_served_from_cache_without_calling_the_client():
    config = Config(model_id="model-a")
    client = make_client("first response")

    outcome1 = get_cached_or_generate(
        client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}]
    )
    outcome2 = get_cached_or_generate(
        client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}]
    )

    assert outcome1.content == "first response"
    assert outcome1.was_cached is False
    assert outcome2.content == "first response"
    assert outcome2.was_cached is True
    client.chat_complete.assert_called_once()


def test_no_cache_flag_always_calls_the_client():
    config = Config(model_id="model-a")
    client = make_client("fresh response")

    get_cached_or_generate(
        client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}], use_cache=False
    )
    outcome = get_cached_or_generate(
        client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}], use_cache=False
    )

    assert outcome.content == "fresh response"
    assert outcome.was_cached is False
    assert client.chat_complete.call_count == 2


def test_cache_key_differs_by_model_system_prompt_and_messages():
    client = make_client("response")
    base_config = Config(model_id="model-a")
    other_model_config = Config(model_id="model-b")

    get_cached_or_generate(client, base_config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}])
    get_cached_or_generate(
        client, other_model_config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}]
    )
    get_cached_or_generate(client, base_config, system_prompt="different sys", messages=[{"role": "user", "content": "hi"}])
    get_cached_or_generate(client, base_config, system_prompt="sys", messages=[{"role": "user", "content": "different"}])

    # None of these four calls should have hit the cache - every key differs.
    assert client.chat_complete.call_count == 4


def test_truncated_responses_are_flagged_and_never_cached():
    """A response cut off by the token limit must not poison the cache."""
    config = Config(model_id="model-a")
    client = make_client('{"concepts": [{"name": "half', finish_reason="length")
    messages = [{"role": "user", "content": "hi"}]

    first = get_cached_or_generate(client, config, system_prompt="sys", messages=messages)
    assert first.truncated is True
    assert first.finish_reason == "length"

    second = get_cached_or_generate(client, config, system_prompt="sys", messages=messages)
    assert second.was_cached is False, "truncated content must never be served from cache"
    assert client.chat_complete.call_count == 2


def test_cache_key_changes_with_generation_settings():
    """Different sampling settings must not share a cached response."""
    client = make_client("response")
    base = Config(model_id="model-a", max_tokens=2048)
    bigger = Config(model_id="model-a", max_tokens=4096)
    hotter = Config(model_id="model-a", temperature=0.9)

    for config in (base, bigger, hotter):
        get_cached_or_generate(client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}])

    assert client.chat_complete.call_count == 3


def test_legacy_cache_entry_without_metadata_is_still_readable():
    """Cache files written before finish_reason existed must keep working."""
    config = Config(model_id="model-a")
    key = _cache_key(config, "sys", [{"role": "user", "content": "hi"}])
    get_cache_dir().mkdir(parents=True, exist_ok=True)
    (get_cache_dir() / f"{key}.json").write_text(json.dumps({"content": "legacy body"}), encoding="utf-8")

    client = make_client("fresh response")
    outcome = get_cached_or_generate(
        client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}]
    )

    assert outcome.content == "legacy body"
    assert outcome.was_cached is True
    assert outcome.finish_reason is None
    client.chat_complete.assert_not_called()
