"""Tests for app settings and JSON logging (Step 0.1)."""

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.logging import JsonFormatter
from app.core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any real env vars for settings fields so tests see only what they set."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()


def test_defaults_load_without_env_file() -> None:
    s = Settings(_env_file=None)

    assert s.api_host == "127.0.0.1"
    assert s.api_port == 8080
    assert s.redis_url == "redis://localhost:6380/0"
    assert s.litellm_base_url == "http://localhost:4000"
    assert s.litellm_api_key.get_secret_value() == ""
    assert s.llamacpp_base_url == "http://127.0.0.1:8081"
    assert (s.po_fast, s.po_accurate, s.po_baseline) == ("po-fast", "po-accurate", "po-baseline")
    assert s.llm_concurrency == 32
    assert s.max_reasks == 1
    assert s.langfuse_host == "http://localhost:3000"
    assert s.langfuse_public_key == ""
    assert s.langfuse_secret_key.get_secret_value() == ""
    assert s.data_dir == Path("data")
    assert s.runs_dir == Path("runs")
    assert s.sqlite_path == Path("data/po_extractor.db")
    assert s.log_level == "INFO"


def test_values_load_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "API_PORT=9090\n"
        "REDIS_URL=redis://redis:6380/1\n"
        "LITELLM_API_KEY=sk-test\n"
        "PO_FAST=my-fast\n"
        "LLM_CONCURRENCY=64\n"
        "MAX_REASKS=0\n"
    )

    s = Settings(_env_file=env_file)

    assert s.api_port == 9090
    assert s.redis_url == "redis://redis:6380/1"
    assert s.litellm_api_key.get_secret_value() == "sk-test"
    assert s.po_fast == "my-fast"
    assert s.llm_concurrency == 64
    assert s.max_reasks == 0


def test_env_var_overrides_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("API_PORT=9090\n")
    monkeypatch.setenv("API_PORT", "7070")

    assert Settings(_env_file=env_file).api_port == 7070


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-very-secret")

    assert "sk-lf-very-secret" not in repr(Settings(_env_file=None))


@pytest.mark.parametrize(("name", "value"), [("LLM_CONCURRENCY", "0"), ("MAX_REASKS", "-1")])
def test_invalid_limits_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.LogRecord("po", logging.INFO, __file__, 1, "job %s done", ("j1",), None)
    record.latency_ms = 812

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "po"
    assert payload["msg"] == "job j1 done"
    assert payload["latency_ms"] == 812
    assert "ts" in payload
