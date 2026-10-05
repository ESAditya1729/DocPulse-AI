"""Configuration and settings management for DocPulse."""

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_CONFIG_DIR = Path.home() / ".docpulse"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"


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

    return Config(**data)


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
