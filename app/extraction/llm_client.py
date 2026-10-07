"""LLM client: one chat-completions call to LiteLLM, never raising for model or network trouble.

The app calls only LiteLLM, by model alias (po-fast, po-accurate, po-baseline, po-moe). The
request parameters of each alias (thinking on/off, penalties, vLLM priority) come from the
active profile set in config/llm_profiles.yaml; nothing model-specific is written here.

Every call:
- is constrained by a JSON schema (response_format json_schema, strict);
- has a max_tokens safety cap (the caller passes settings.llm_max_tokens_header / _lines);
- times out after settings.llm_timeout_s per attempt;
- retries transient errors (timeout, connection error, 429, 5xx) with exponential backoff,
  up to settings.llm_transient_retries times. Validation re-asks are a separate mechanism
  (Step 3.7). Note: LiteLLM has its own num_retries in infra/litellm/config.*.yaml.

The result always comes back as a CallResult: a failed request, invalid JSON or a truncated
answer sets `error` instead of raising. Short keys of the header schema are mapped back to
PurchaseOrder field names; other keys (the line-item "rows") are kept as they are.
"""

import asyncio
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
import openai

from app.core.logging import get_logger
from app.core.settings import Settings
from schema.llm_keys import FULL_KEYS
from schema.llm_schemas import parse_llm_json

log = get_logger(__name__)

ErrorKind = Literal["transient", "api", "truncated", "invalid_json"]


@dataclass(frozen=True)
class Tokens:
    """Token counts of one call, as reported by the server."""

    prompt: int = 0
    completion: int = 0  # all generated tokens, reasoning included
    reasoning: int | None = None  # None when the server does not report it


@dataclass(frozen=True)
class Timings:
    """Wall-clock start and end (epoch seconds, for timelines) and duration of the call."""

    start_ts: float
    end_ts: float
    duration_ms: float  # retries and backoff included


@dataclass(frozen=True)
class CallResult:
    """The outcome of one LLM call. `error` is None on success."""

    call_id: str
    alias: str
    parsed: dict[str, Any] | None  # the JSON object, header keys mapped to full field names
    raw_content: str | None  # the answer text as returned
    reasoning_content: str | None  # thinking text, if any: for debugging only
    tokens: Tokens
    timings: Timings
    attempts: int
    finish_reason: str | None = None
    error: str | None = None
    error_kind: ErrorKind | None = None
    request_body: dict[str, Any] = field(default_factory=dict, repr=False)  # without messages

    @property
    def ok(self) -> bool:
        return self.error is None


_THINK = re.compile(r"^\s*<think>(.*?)</think>", re.DOTALL)


def split_reasoning(message: dict[str, Any]) -> tuple[str, str]:
    """(reasoning text, answer text). Reasoning is read from `reasoning_content`, or from a
    leading <think>...</think> block if the server left it in the content."""
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    if match := _THINK.match(content):
        reasoning = reasoning or match[1].strip()
        content = content[match.end() :]
    return reasoning, content


def to_full_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Map short keys (schema/llm_keys.py) to full field names; other keys stay as they are."""
    return {FULL_KEYS.get(key, key): value for key, value in data.items()}


def is_transient(exc: Exception) -> bool:
    """Timeouts, connection errors, 429 and 5xx are worth retrying; other errors are not."""
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return True
    return isinstance(exc, openai.APIStatusError) and (
        exc.status_code == 429 or exc.status_code >= 500
    )


class LlmClient:
    """Async client for LiteLLM's OpenAI-compatible API. Create one per process and reuse it."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        """`http_client` lets tests inject a mock transport."""
        self._settings = settings
        self._profiles = settings.llm_profiles()
        self._client = openai.AsyncOpenAI(
            base_url=f"{settings.litellm_base_url.rstrip('/')}/v1",
            api_key=settings.litellm_api_key.get_secret_value() or "no-key",
            timeout=settings.llm_timeout_s,
            max_retries=0,  # retries are done here, visibly, with our own backoff
            http_client=http_client,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "LlmClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def request_body(
        self,
        alias: str,
        schema: dict[str, Any],
        max_tokens: int,
        priority: int | None = None,
        schema_name: str = "extraction",
    ) -> dict[str, Any]:
        """The request body for `alias`, without messages. ValueError for an unknown alias."""
        if alias not in self._profiles:
            raise ValueError(
                f"no request profile for alias {alias!r} in set "
                f"{self._settings.llm_profile_set!r} (have {sorted(self._profiles)})"
            )
        profile = self._profiles[alias]
        body: dict[str, Any] = {
            "model": alias,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
            **profile.body,
        }
        if profile.send_priority and priority is not None:
            body["priority"] = priority
        return body

    async def call(
        self,
        alias: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        max_tokens: int,
        priority: int | None = None,
        call_id: str | None = None,
        schema_name: str = "extraction",
    ) -> CallResult:
        """Run one chat-completions call and return its CallResult (never raises for API errors).

        `priority` is sent only when the alias's profile has send_priority (vLLM: smaller is
        served first). Raises ValueError only for a programming error: an unknown alias.
        """
        body = self.request_body(alias, schema, max_tokens, priority, schema_name)
        call_id = call_id or uuid.uuid4().hex[:12]
        start_ts, start = time.time(), time.perf_counter()

        def result(attempts: int, **fields: Any) -> CallResult:
            timings = Timings(start_ts, time.time(), round((time.perf_counter() - start) * 1000, 1))
            fields.setdefault("parsed", None)
            fields.setdefault("raw_content", None)
            fields.setdefault("reasoning_content", None)
            fields.setdefault("tokens", Tokens())
            return CallResult(
                call_id, alias, timings=timings, attempts=attempts, request_body=body, **fields
            )

        attempts = 0
        while True:
            attempts += 1
            try:
                response = await self._client.chat.completions.create(
                    model=alias,
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format=body["response_format"],
                    extra_body={k: v for k, v in body.items() if k not in _TYPED_FIELDS},
                    timeout=self._settings.llm_timeout_s,
                )
                break
            except openai.APIError as exc:
                transient = is_transient(exc)
                error = f"{type(exc).__name__}: {exc}"
                if transient and attempts <= self._settings.llm_transient_retries:
                    delay = self._settings.llm_backoff_s * 2 ** (attempts - 1)
                    delay *= 1 + random.random() / 4  # jitter: retries of a burst spread out
                    log.warning(
                        "transient LLM error, retrying",
                        extra={
                            "call_id": call_id,
                            "alias": alias,
                            "attempt": attempts,
                            "error": type(exc).__name__,
                            "delay_s": round(delay, 2),
                        },
                    )
                    await asyncio.sleep(delay)
                    continue
                return result(attempts, error=error, error_kind="transient" if transient else "api")

        data = response.model_dump()
        choice = data["choices"][0] if data.get("choices") else {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        reasoning, content = split_reasoning(message)
        usage = data.get("usage") or {}
        tokens = Tokens(
            prompt=int(usage.get("prompt_tokens") or 0),
            completion=int(usage.get("completion_tokens") or 0),
            reasoning=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        )
        answer = {
            "raw_content": content,
            "reasoning_content": reasoning or None,
            "tokens": tokens,
            "finish_reason": finish_reason,
        }
        if finish_reason == "length":
            error = f"truncated at max_tokens={max_tokens} ({len(content)} chars)"
            return result(attempts, error=error, error_kind="truncated", **answer)
        try:
            parsed = parse_llm_json(content)
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            error = f"invalid JSON: {exc} (finish_reason={finish_reason}, {len(content)} chars)"
            return result(attempts, error=error, error_kind="invalid_json", **answer)
        if not isinstance(parsed, dict):
            error = f"expected a JSON object, got {type(parsed).__name__}"
            return result(attempts, error=error, error_kind="invalid_json", **answer)
        return result(attempts, parsed=to_full_keys(parsed), **answer)


# Fields passed to the OpenAI SDK as typed arguments; everything else goes in extra_body.
_TYPED_FIELDS = frozenset({"model", "max_tokens", "response_format"})
