"""Configuration and settings management for DocPulse."""

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from docpulse.errors import ConfigError

DEFAULT_CONFIG_DIR = Path.home() / ".docpulse"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"

SETUP_HINT = "Run 'docpulse init' to configure the gateway, or set environment variables:\n  DOCPULSE_ENDPOINT_URL / DOCPULSE_API_KEY / DOCPULSE_NAMESPACE / DOCPULSE_MODEL_ID"


class Config(BaseModel):
    """DocPulse runtime configuration."""

    endpoint_url: str = Field(
        default="",
        description="IBM ICA Gateway endpoint URL (set via 'docpulse init' or DOCPULSE_ENDPOINT_URL/ICA_ENDPOINT_URL)",
    )
    api_key: str = Field(
        default="",
        description="IBM ICA Gateway API Key",
    )
    namespace: str = Field(
        default="chat-models",
        description="ICA Namespace (chat-models, assistants, agents, digital-workforce)",
    )
    model_id: str = Field(
        default="ibm/granite-3-8b-instruct",
        description="Default LLM model identifier",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM generation temperature",
    )
    max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum response tokens",
    )
    request_timeout: float = Field(
        default=120.0,
        gt=0,
        description="Seconds to wait for a gateway response before giving up",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="How many times to retry a failed or throttled gateway request",
    )
    retry_backoff: float = Field(
        default=0.8,
        ge=0.0,
        description="Base seconds for exponential retry backoff (jitter is added on top)",
    )

    def missing_fields(self) -> list[str]:
        """Names of the settings that still need a value before any LLM call."""
        return [name for name in ("endpoint_url", "model_id") if not getattr(self, name).strip()]

    def validate_for_command(self) -> None:
        """Raise ConfigError with recovery instructions if the config is unusable.

        An API key is *not* required - some gateways are keyless - so a missing
        key is reported by the caller as a warning rather than blocking here.
        """
        missing = self.missing_fields()
        if missing:
            raise ConfigError(
                "DocPulse is not configured: missing " + ", ".join(missing) + ".",
                hint=SETUP_HINT,
            )
        if not self.endpoint_url.startswith(("http://", "https://")):
            raise ConfigError(
                f"Configured endpoint_url is not a valid URL: {self.endpoint_url!r}",
                hint=SETUP_HINT,
            )

    def redacted(self) -> dict[str, Any]:
        """Config as a dict with the API key masked, safe to print or log."""
        data = self.model_dump()
        key = data.get("api_key") or ""
        if key:
            data["api_key"] = f"{key[:4]}...{key[-2:]}" if len(key) > 8 else "***"
        return data


def get_config_path() -> Path:
    """Return the configuration file path, allowing override via env var."""
    custom_path = os.getenv("DOCPULSE_CONFIG_PATH")
    if custom_path:
        return Path(custom_path)
    return DEFAULT_CONFIG_FILE


def load_config() -> Config:
    """Load configuration from disk or environment variables."""
    config_file = get_config_path()
    data = {}

    if config_file.exists():
        try:
            with open(config_file, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}

    # Environment variables take precedence if present
    env_endpoint = os.getenv("DOCPULSE_ENDPOINT_URL") or os.getenv("ICA_ENDPOINT_URL")
    env_api_key = os.getenv("DOCPULSE_API_KEY") or os.getenv("ICA_API_KEY")
    env_namespace = os.getenv("DOCPULSE_NAMESPACE") or os.getenv("ICA_NAMESPACE")
    env_model = os.getenv("DOCPULSE_MODEL_ID") or os.getenv("ICA_MODEL_ID")

    if env_endpoint:
        data["endpoint_url"] = env_endpoint
    if env_api_key:
        data["api_key"] = env_api_key
    if env_namespace:
        data["namespace"] = env_namespace
    if env_model:
        data["model_id"] = env_model

    # Transport tuning. Invalid values fall back to the defaults instead of
    # raising, so a typo in an env var can't make every command unusable.
    env_timeout = _env_float("DOCPULSE_TIMEOUT")
    if env_timeout is not None and env_timeout > 0:
        data["request_timeout"] = env_timeout
    env_retries = _env_int("DOCPULSE_MAX_RETRIES")
    if env_retries is not None and env_retries >= 0:
        data["max_retries"] = env_retries
    env_backoff = _env_float("DOCPULSE_RETRY_BACKOFF")
    if env_backoff is not None and env_backoff >= 0:
        data["retry_backoff"] = env_backoff

    return Config(**data)


def _env_float(name: str) -> float | None:
    """Read an env var as a float, returning None when unset or unparseable."""
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    """Read an env var as an int, returning None when unset or unparseable."""
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def save_config(config: Config) -> Path:
    """Save configuration to disk."""
    config_file = get_config_path()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)
    # Config contains the API key; restrict to owner read/write (no-op on Windows).
    try:
        os.chmod(config_file, 0o600)
    except OSError:
        pass
    return config_file
