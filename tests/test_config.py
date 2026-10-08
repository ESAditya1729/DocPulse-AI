"""Tests for configuration validation, redaction, and transport settings."""

import json

import pytest

from docpulse.config import Config, load_config, save_config
from docpulse.errors import ConfigError


def test_validate_passes_with_only_an_endpoint():
    """An API key is optional so keyless gateways keep working."""
    Config(endpoint_url="https://gateway.example.com/v1").validate_for_command()


def test_validate_rejects_missing_endpoint_with_recovery_instructions():
    with pytest.raises(ConfigError) as excinfo:
        Config().validate_for_command()

    message = excinfo.value.user_message()
    assert "endpoint_url" in message
    assert "docpulse init" in message
    assert excinfo.value.exit_code == 2


def test_validate_rejects_an_endpoint_without_a_scheme():
    with pytest.raises(ConfigError, match="not a valid URL"):
        Config(endpoint_url="gateway.example.com/v1").validate_for_command()


def test_missing_fields_lists_only_what_is_unset():
    assert Config(endpoint_url="https://x.example.com").missing_fields() == []
    assert "endpoint_url" in Config().missing_fields()


def test_redacted_masks_the_api_key():
    redacted = Config(api_key="super-secret-key-value").redacted()

    assert "super-secret-key-value" not in json.dumps(redacted)
    # Long keys keep just enough to identify which key is configured.
    assert redacted["api_key"] == "supe...ue"


def test_redacted_handles_a_short_key():
    assert Config(api_key="abc").redacted()["api_key"] == "***"


def test_redacted_handles_no_key():
    assert Config(api_key="").redacted()["api_key"] == ""


def test_env_overrides_transport_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setenv("DOCPULSE_TIMEOUT", "30")
    monkeypatch.setenv("DOCPULSE_MAX_RETRIES", "7")
    monkeypatch.setenv("DOCPULSE_RETRY_BACKOFF", "0.25")

    config = load_config()

    assert config.request_timeout == 30.0
    assert config.max_retries == 7
    assert config.retry_backoff == 0.25


@pytest.mark.parametrize(
    ("env_var", "value"),
    [("DOCPULSE_TIMEOUT", "not-a-number"), ("DOCPULSE_MAX_RETRIES", "many"), ("DOCPULSE_RETRY_BACKOFF", "")],
)
def test_unparseable_env_values_fall_back_to_defaults(monkeypatch, tmp_path, env_var, value):
    """A typo in an env var must not make every command unusable."""
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setenv(env_var, value)

    config = load_config()

    assert config.request_timeout == Config().request_timeout
    assert config.max_retries == Config().max_retries


def test_saved_config_round_trips_transport_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "config.json"))

    save_config(Config(endpoint_url="https://x.example.com", request_timeout=15.0, max_retries=1))

    loaded = load_config()
    assert loaded.request_timeout == 15.0
    assert loaded.max_retries == 1


def test_corrupt_config_file_falls_back_to_defaults(tmp_path, monkeypatch):
    bad = tmp_path / "config.json"
    bad.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(bad))

    assert load_config().endpoint_url == ""