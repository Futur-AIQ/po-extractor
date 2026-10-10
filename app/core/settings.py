"""Application settings, loaded from environment variables and `.env`.

Every URL, port, key and model alias used by the app lives here. Defaults match the
local Mac dev setup (see the Ports section of CLAUDE.md).
"""

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Request fields the LLM client always sets itself; a profile may not override them.
RESERVED_REQUEST_FIELDS = frozenset(
    {"model", "messages", "max_tokens", "response_format", "stream", "priority"}
)


class LlmProfile(BaseModel):
    """Request parameters for one model alias (config/llm_profiles.yaml)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    body: dict[str, Any] = Field(default_factory=dict)  # added to the request body as-is
    send_priority: bool = False  # send vLLM's per-request "priority"

    @field_validator("body")
    @classmethod
    def _no_reserved_fields(cls, body: dict[str, Any]) -> dict[str, Any]:
        reserved = sorted(RESERVED_REQUEST_FIELDS & body.keys())
        if reserved:
            raise ValueError(f"set by the client, not allowed in a profile: {reserved}")
        return body


def load_llm_profiles(path: Path, profile_set: str) -> dict[str, LlmProfile]:
    """The request profiles of one set (`dev`, `h100`), keyed by model alias."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sets = data.get("profile_sets") or {}
    if profile_set not in sets:
        raise ValueError(f"profile set {profile_set!r} not in {path} (have {sorted(sets)})")
    # Deep copy: YAML anchors make profiles share nested dicts.
    profiles = copy.deepcopy(sets[profile_set])
    return {alias: LlmProfile.model_validate(profile) for alias, profile in profiles.items()}


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
    po_moe: str = "po-moe"  # optional MoE candidate (H100 only)

    # LLM requests (app/extraction/llm_client.py). Per-alias parameters (thinking, penalties,
    # priority) come from the active set in config/llm_profiles.yaml.
    llm_profile_set: Literal["dev", "h100"] = "dev"
    llm_profiles_path: Path = PROJECT_ROOT / "config" / "llm_profiles.yaml"
    llm_timeout_s: float = Field(default=120.0, gt=0, description="Timeout per LLM request")
    llm_transient_retries: int = Field(
        default=2, ge=0, description="Retries on timeout, connection error, 429 and 5xx"
    )
    llm_backoff_s: float = Field(default=1.0, ge=0, description="First retry delay; doubles")
    llm_max_tokens_header: int = Field(default=2000, gt=0, description="Cap for header calls")
    llm_max_tokens_lines: int = Field(default=6000, gt=0, description="Cap for line-item calls")

    # Call planning (app/extraction/planner.py, PRD FR-07): two_call = header + line items,
    # per_page = header + one line-item call per item page, adaptive = two_call unless the PO
    # has more than adaptive_row_threshold estimated rows or adaptive_page_threshold item pages.
    call_strategy: Literal["two_call", "per_page", "adaptive"] = "adaptive"
    adaptive_row_threshold: int = Field(default=40, ge=0)
    adaptive_page_threshold: int = Field(default=2, ge=0)

    # Extraction limits. llm_concurrency caps in-flight LLM calls per process; keep it at or
    # below vLLM --max-num-seqs, and never set LiteLLM limits lower than it (PRD FR-15).
    llm_concurrency: int = Field(default=32, gt=0, description="Global cap on in-flight LLM calls")
    max_reasks: int = Field(default=1, ge=0, description="Targeted re-asks per PO on failure")
    # A retry starts only before this many seconds since the PO started, so a retry never
    # pushes a PO past the 60 s ceiling (PRD §3); later failures go to review instead.
    retry_deadline_s: float = Field(default=40.0, ge=0)
    # Extra max_tokens for a retry call on po-accurate: its thinking counts as output tokens.
    llm_retry_reasoning_tokens: int = Field(default=4000, ge=0)

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
    # Mock ERP master data for validation rule 10 (parties, item codes, processed PO numbers).
    masters_dir: Path = Path("data/synthetic/masters")

    # Logging
    log_level: str = "INFO"

    def llm_profiles(self) -> dict[str, LlmProfile]:
        """Request profiles of the active set (LLM_PROFILE_SET), keyed by model alias."""
        return load_llm_profiles(self.llm_profiles_path, self.llm_profile_set)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (loaded once)."""
    return Settings()
