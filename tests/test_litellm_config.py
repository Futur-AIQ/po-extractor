"""Checks that the LiteLLM configs expose exactly the aliases the app uses (Step 0.4)."""

from pathlib import Path

import pytest
import yaml

from app.core.settings import Settings

CONFIG_DIR = Path(__file__).resolve().parents[1] / "infra" / "litellm"


def load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


@pytest.mark.parametrize("name", ["config.dev.yaml", "config.h100.yaml"])
def test_config_defines_app_aliases(name: str) -> None:
    settings = Settings(_env_file=None)
    config = load(name)

    aliases = {m["model_name"] for m in config["model_list"]}

    assert aliases == {settings.po_fast, settings.po_accurate, settings.po_baseline}
    assert config["litellm_settings"]["num_retries"] == 2
    assert config["litellm_settings"]["request_timeout"] > 0
    # Proxy key comes from the environment, never from the file.
    assert config["general_settings"]["master_key"] == "os.environ/LITELLM_API_KEY"


def test_dev_config_routes_everything_to_local_llamacpp() -> None:
    for model in load("config.dev.yaml")["model_list"]:
        assert model["litellm_params"]["api_base"] == "http://127.0.0.1:8081/v1"


@pytest.mark.parametrize("name", ["config.dev.yaml", "config.h100.yaml"])
def test_baseline_alias_allows_long_requests(name: str) -> None:
    models = {m["model_name"]: m["litellm_params"] for m in load(name)["model_list"]}
    assert models["po-baseline"]["timeout"] >= 600  # baseline default --timeout
