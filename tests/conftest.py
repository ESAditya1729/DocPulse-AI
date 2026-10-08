"""Shared pytest fixtures."""

import pytest

from docpulse.config import Config, save_config


@pytest.fixture(autouse=True)
def isolate_docpulse_state(tmp_path, monkeypatch):
    """Give every test an isolated cache and a config with a usable endpoint.

    Commands now refuse to run against an unconfigured DocPulse, so tests get a
    pre-written config pointing at a fake gateway. Both paths live under
    tmp_path so no test can read or write the developer's real ~/.docpulse.
    """
    monkeypatch.setenv("DOCPULSE_CACHE_DIR", str(tmp_path / "cache"))

    # The path must be redirected BEFORE save_config runs, otherwise it writes
    # to the developer's real ~/.docpulse/config.json.
    config_path = tmp_path / "config.json"
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(config_path))
    save_config(
        Config(
            endpoint_url="https://gateway.example.com/v1",
            api_key="test-api-key",
            namespace="chat-models",
            model_id="test-model",
        )
    )
    return config_path