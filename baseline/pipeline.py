"""Baseline B0: a faithful reproduction of the CURRENT production approach (PRD §1, §12 row B0).

Improvements are measured against this, so it deliberately keeps today's choices:
- every page rendered with PyMuPDF at 120 dpi and sent as an image, all in ONE request;
  no text layer;
- the full Pydantic JSON schema as guided JSON: full field names, every field including the
  computed ones, line items as objects (PurchaseOrder.model_json_schema());
- model alias po-baseline with thinking at the model's default (enable_thinking is not sent)
  and no temperature override unless asked;
- a large max_tokens (16000) and a long timeout (600 s).

Like the app, it calls only LiteLLM. Failures never raise: an HTTP error or timeout gives a
'failed' result, malformed or truncated JSON gives a 'parse_error' result.
"""

import asyncio
import base64
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import pymupdf
from pydantic import ValidationError

from app.core.settings import Settings
from eval.run_format import Status, TokenUsage
from schema.llm_schemas import parse_llm_json
from schema.po_schema import PurchaseOrder

DEFAULT_DPI = 120
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT_S = 600.0

SYSTEM_PROMPT = (
    "You are an expert at extracting data from Indian GST purchase orders. "
    "Read the purchase order page images and return all its data as JSON matching the schema."
)
USER_PROMPT = "Extract the purchase order from these {pages} page images."


@dataclass(frozen=True)
class BaselineSettings:
    """How the baseline calls the model. Defaults reproduce today's production setup."""

    litellm_base_url: str
    litellm_api_key: str = field(repr=False)
    model_alias: str
    dpi: int = DEFAULT_DPI
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_s: float = DEFAULT_TIMEOUT_S
    temperature: float | None = None  # None: not sent, the model's default applies

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> "BaselineSettings":
        """LiteLLM URL, key and the po-baseline alias from app settings, plus overrides."""
        base = cls(
            litellm_base_url=settings.litellm_base_url,
            litellm_api_key=settings.litellm_api_key.get_secret_value(),
            model_alias=settings.po_baseline,
        )
        return replace(base, **overrides)


@dataclass
class BaselineResult:
    """Outcome of one baseline extraction."""

    po: PurchaseOrder | None
    status: Status  # completed, parse_error or failed
    error: str | None
    raw: dict[str, Any]  # request (without image data) and the full response, reasoning included
    timings_ms: dict[str, float]  # render, llm, parse, total
    tokens: TokenUsage
    calls: int = 1
    # Well-formed JSON that failed PurchaseOrder validation (e.g. "quantity": "63 NOS"). It is a
    # parse error, as in production, but its values are kept so accuracy can still be scored.
    invalid_data: dict[str, Any] | None = None


# =========================================================================================
# Steps
# =========================================================================================


def render_pages(pdf_path: Path, dpi: int = DEFAULT_DPI) -> list[bytes]:
    """Every page of the PDF as a PNG at `dpi` (CPU work: call via asyncio.to_thread)."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_pixmap(dpi=dpi).tobytes("png") for page in doc]


def build_request(images: list[bytes], cfg: BaselineSettings) -> dict[str, Any]:
    """The single chat-completions request: page images + the full PurchaseOrder schema."""
    image_parts = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
        }
        for png in images
    ]
    body: dict[str, Any] = {
        "model": cfg.model_alias,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT.format(pages=len(images))},
                    *image_parts,
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "purchase_order",
                "schema": PurchaseOrder.model_json_schema(),
                "strict": True,
            },
        },
    }
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    return body


def _without_image_data(body: dict[str, Any]) -> dict[str, Any]:
    """The request as stored in raw/: image data replaced by its size, schema left out."""
    messages = []
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, list):
            content = [
                {"type": "image_url", "image_url": f"<png, {len(part['image_url']['url'])} chars>"}
                if part["type"] == "image_url"
                else part
                for part in content
            ]
        messages.append({**message, "content": content})
    return {k: v for k, v in body.items() if k != "response_format"} | {
        "messages": messages,
        "response_format": "json_schema purchase_order (PurchaseOrder.model_json_schema())",
    }


_THINK = re.compile(r"^\s*<think>(.*?)</think>", re.DOTALL)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def split_reasoning(message: dict[str, Any]) -> tuple[str, str]:
    """(reasoning text, answer text). Reasoning is read from `reasoning_content`, or from a
    leading <think>...</think> block if the server left it in the content."""
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    if match := _THINK.match(content):
        reasoning = reasoning or match[1].strip()
        content = content[match.end() :]
    return reasoning, content


def token_usage(usage: dict[str, Any], message: dict[str, Any]) -> TokenUsage:
    """Token counts of the response.

    Reasoning tokens come from usage.completion_tokens_details when the server reports them.
    Otherwise (llama.cpp reports none) they are estimated from the reasoning text's share of
    the generated characters, and flagged as an estimate.
    """
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    reported = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    reasoning, content = split_reasoning(message)
    if reported is not None and (reported > 0 or not reasoning):
        return TokenUsage(prompt=prompt, completion=completion, reasoning=int(reported))
    if reasoning:
        share = len(reasoning) / (len(reasoning) + len(content))
        return TokenUsage(
            prompt=prompt,
            completion=completion,
            reasoning=round(completion * share),
            reasoning_estimated=True,
        )
    return TokenUsage(prompt=prompt, completion=completion)


def parse_po(
    content: str, finish_reason: str | None
) -> tuple[PurchaseOrder | None, dict[str, Any] | None, str | None]:
    """(po, invalid_data, error). Never raises.

    - valid PurchaseOrder:                 (po, None, None)
    - JSON object that fails validation:   (None, the parsed object, error)
    - empty, malformed or truncated JSON:  (None, None, error)
    """
    if not content.strip():
        return None, None, f"empty content (finish_reason={finish_reason})"
    if match := _FENCE.match(content):
        content = match[1]
    try:
        data = parse_llm_json(content)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        detail = f"finish_reason={finish_reason}, {len(content)} chars"
        return None, None, f"invalid JSON: {exc} ({detail})"
    try:
        return PurchaseOrder.model_validate(data), None, None
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        error = f"schema validation: {exc.error_count()} errors, first {where}: {first['msg']}"
        return None, (data if isinstance(data, dict) else None), error


# =========================================================================================
# One document
# =========================================================================================


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def extract_baseline(
    pdf_path: Path, cfg: BaselineSettings, client: httpx.AsyncClient | None = None
) -> BaselineResult:
    """Run the baseline on one PDF. Pass a shared `client` when running many documents."""
    start = time.perf_counter()
    images = await asyncio.to_thread(render_pages, pdf_path, cfg.dpi)
    timings = {"render": _ms(start)}
    body = build_request(images, cfg)
    raw: dict[str, Any] = {"request": _without_image_data(body), "pages": len(images)}

    def finish(status: Status, error: str | None, tokens: TokenUsage | None = None,
               po: PurchaseOrder | None = None,
               invalid_data: dict[str, Any] | None = None) -> BaselineResult:  # fmt: skip
        timings.setdefault("parse", 0.0)
        timings["total"] = _ms(start)
        raw["error"] = error
        tokens = tokens or TokenUsage()
        return BaselineResult(po, status, error, raw, timings, tokens, 1, invalid_data)

    url = f"{cfg.litellm_base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.litellm_api_key}"}
    llm_start = time.perf_counter()
    try:
        if client is None:
            async with httpx.AsyncClient() as own_client:
                response = await own_client.post(
                    url, json=body, headers=headers, timeout=cfg.timeout_s
                )
        else:
            response = await client.post(url, json=body, headers=headers, timeout=cfg.timeout_s)
    except httpx.HTTPError as exc:
        timings["llm"] = _ms(llm_start)
        return finish("failed", f"{type(exc).__name__}: {exc}")
    timings["llm"] = _ms(llm_start)
    raw["status_code"] = response.status_code
    if response.status_code != 200:
        raw["response_text"] = response.text
        return finish("failed", f"HTTP {response.status_code}: {response.text[:300]}")
    try:
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raw["response_text"] = response.text
        return finish("failed", f"unexpected response: {exc!r}")
    raw["response"] = data  # includes message.reasoning_content (reasoning stays in raw only)

    parse_start = time.perf_counter()
    tokens = token_usage(data.get("usage") or {}, message)
    _, content = split_reasoning(message)
    po, invalid_data, error = parse_po(content, choice.get("finish_reason"))
    timings["parse"] = _ms(parse_start)
    if po is None:
        return finish("parse_error", error, tokens, invalid_data=invalid_data)
    return finish("completed", None, tokens, po)
