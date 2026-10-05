"""Tests for the local content-addressed response cache."""

from unittest.mock import MagicMock

from docpulse.cache import get_cache_dir, get_cached_or_generate
from docpulse.config import Config
from docpulse.llm import LLMResponse


def make_client(content: str) -> MagicMock:
    client = MagicMock()
    client.chat_complete.return_value = LLMResponse(content=content, model="test-model")
    return client


def test_cache_dir_respects_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "my-cache"
    monkeypatch.setenv("DOCPULSE_CACHE_DIR", str(custom))
    assert get_cache_dir() == custom


def test_second_call_is_served_from_cache_without_calling_the_client():
    config = Config(model_id="model-a")
    client = make_client("first response")

    content1, cached1 = get_cached_or_generate(
        client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}]
    )
    content2, cached2 = get_cached_or_generate(
        client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}]
    )

    assert content1 == "first response"
    assert cached1 is False
    assert content2 == "first response"
    assert cached2 is True
    client.chat_complete.assert_called_once()


def test_no_cache_flag_always_calls_the_client():
    config = Config(model_id="model-a")
    client = make_client("fresh response")

    get_cached_or_generate(
        client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}], use_cache=False
    )
    content, cached = get_cached_or_generate(
        client, config, system_prompt="sys", messages=[{"role": "user", "content": "hi"}], use_cache=False
    )

    assert content == "fresh response"
    assert cached is False
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
