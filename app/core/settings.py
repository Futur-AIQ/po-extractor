"""Application settings, loaded from environment variables and `.env`.

Every URL, port, key and model alias used by the app lives here. Defaults match the
local Mac dev setup (see the Ports section of CLAUDE.md).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for the API, worker, LLM gateway, tracing and storage."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8080, ge=1, le=65535)

    # App Redis (job queue). Port 6380 avoids clashing with Langfuse's own Redis.
    redis_url: str = "redis://localhost:6380/0"

    # LiteLLM proxy: the only LLM endpoint the app talks to.
    litellm_base_url: str = "http://localhost:4000"
    litellm_api_key: SecretStr = SecretStr("")

    # Local llama.cpp server. Used only by dev tooling (`make doctor`); the app never calls it.
    llamacpp_base_url: str = "http://127.0.0.1:8081"

    # Model aliases defined in the LiteLLM config.
    po_fast: str = "po-fast"
    po_accurate: str = "po-accurate"
    po_baseline: str = "po-baseline"

    # Extraction limits
    llm_concurrency: int = Field(default=32, gt=0, description="Global cap on in-flight LLM calls")
    max_reasks: int = Field(default=1, ge=0, description="Targeted re-asks per PO on failure")

    # Pre-processing (app/extraction/preprocess.py)
    native_min_chars: int = Field(
        default=50, ge=0, description="A page is native above this many letters/digits of text"
    )
    image_dpi: int = Field(default=150, gt=0, description="Page image resolution")
    image_format: Literal["jpeg", "png"] = "jpeg"
    image_jpeg_quality: int = Field(default=92, ge=1, le=100)
    max_image_px: int = Field(default=2400, gt=0, description="Cap on a page image's longest side")
    skip_fieldless_pages: bool = True  # skip native pages that are only terms and conditions
    # What a native page sends: its text layer and image (default), or one of them (ablations).
    native_mode: Literal["text+image", "text", "image"] = "text+image"
    # Text layer of native pages: PyMuPDF plain text (default) or pymupdf4llm markdown.
    text_engine: Literal["pymupdf", "pymupdf4llm"] = "pymupdf"

    # Langfuse (self-hosted)
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")

    # Storage
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")  # extraction runs for evaluation (eval/run_format.py)
    sqlite_path: Path = Path("data/po_extractor.db")

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (loaded once)."""
    return Settings()
