"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolate_cache_dir(tmp_path, monkeypatch):
    """Give every test an empty, per-test response cache instead of the real ~/.docpulse/cache."""
    monkeypatch.setenv("DOCPULSE_CACHE_DIR", str(tmp_path / "cache"))
