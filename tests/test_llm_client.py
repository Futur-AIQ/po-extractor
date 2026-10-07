"""Tests for the LLM client and request profiles (Step 3.4). Mocked HTTP transport, no network."""

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from app.core.settings import Settings, load_llm_profiles
from app.extraction.llm_client import LlmClient, Tokens

SCHEMA = {"type": "object", "properties": {"po_no": {"type": "string"}}}
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "page"}]
ALIASES = ("po-fast", "po-accurate", "po-baseline", "po-moe")


def settings(profile_set: str = "h100", **overrides) -> Settings:
    values = {"litellm_api_key": "sk-test", "llm_backoff_s": 0, **overrides}  # no real waits
    return Settings(_env_file=None, llm_profile_set=profile_set, **values)


def completion(
    content: str = '{"po_no": "PO/1"}',
    finish_reason: str = "stop",
    reasoning: str | None = None,
    usage: dict | None = None,
) -> dict:
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "po-fast",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230},
    }


class Server:
    """A fake LiteLLM: replies from a list of handlers, one per request, and records requests."""

    def __init__(self, *replies: Callable[[httpx.Request], httpx.Response] | dict | int) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if callable(reply):
            return reply(request)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": f"status {reply}"}})
        return httpx.Response(200, json=reply)

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


def deterministic(body: dict) -> bool:
    """FR-08 decoding parameters."""
    params = ("temperature", "presence_penalty", "repetition_penalty")
    return [body[name] for name in params] == [0, 0, 1.0]


def client(server: Server, s: Settings | None = None) -> LlmClient:
    transport = httpx.MockTransport(server)
    return LlmClient(s or settings(), http_client=httpx.AsyncClient(transport=transport))


# --- Profiles ----------------------------------------------------------------------------


def test_both_profile_sets_define_every_alias() -> None:
    for profile_set in ("dev", "h100"):
        assert set(settings(profile_set).llm_profiles()) == set(ALIASES)


def test_thinking_low_shape_is_defined_once_and_marked_verify() -> None:
    text = Path(settings().llm_profiles_path).read_text(encoding="utf-8")
    assert text.count("reasoning_effort") == 1
    assert "VERIFY against the Qwen3.8 model card / vLLM version in Phase 7" in text


def test_unknown_profile_set_and_reserved_fields_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="profile set 'prod'"):
        load_llm_profiles(settings().llm_profiles_path, "prod")
    path = tmp_path / "profiles.yaml"
    path.write_text("profile_sets:\n  dev:\n    po-fast:\n      body: {max_tokens: 99}\n")
    with pytest.raises(ValidationError, match="not allowed in a profile"):
        load_llm_profiles(path, "dev")
    path.write_text("profile_sets:\n  dev:\n    po-fast:\n      thinking: true\n")
    with pytest.raises(ValidationError):  # unknown profile key
        load_llm_profiles(path, "dev")


# --- Request body per profile ------------------------------------------------------------


async def test_h100_po_fast_body() -> None:
    server = Server(completion())
    async with client(server) as llm:
        result = await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=2000, priority=7)
    assert result.ok
    body = server.body()
    assert body["model"] == "po-fast" and body["messages"] == MESSAGES
    assert body["max_tokens"] == 2000
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "schema": SCHEMA, "strict": True},
    }
    assert deterministic(body)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["priority"] == 7
    assert "reasoning_effort" not in body
    assert server.requests[0].url == "http://localhost:4000/v1/chat/completions"
    assert server.requests[0].headers["authorization"] == "Bearer sk-test"


async def test_h100_po_accurate_thinks_with_low_effort() -> None:
    server = Server(completion())
    async with client(server) as llm:
        await llm.call("po-accurate", MESSAGES, SCHEMA, max_tokens=6000, priority=3)
    body = server.body()
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["reasoning_effort"] == "low"
    assert deterministic(body)
    assert body["priority"] == 3


async def test_dev_profiles_never_send_priority_and_have_no_effort_levels() -> None:
    server = Server(completion())
    async with client(server, settings("dev")) as llm:
        await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=2000, priority=7)
        await llm.call("po-accurate", MESSAGES, SCHEMA, max_tokens=2000, priority=7)
    fast, accurate = server.body(0), server.body(1)
    assert fast["chat_template_kwargs"] == {"enable_thinking": False}
    assert accurate["chat_template_kwargs"] == {"enable_thinking": True}
    assert "priority" not in fast and "priority" not in accurate
    assert "reasoning_effort" not in accurate
    assert fast["repetition_penalty"] == accurate["repetition_penalty"] == 1.0


async def test_no_priority_field_without_a_priority_and_baseline_uses_model_defaults() -> None:
    server = Server(completion())
    async with client(server) as llm:
        await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=2000)
        await llm.call("po-baseline", MESSAGES, SCHEMA, max_tokens=2000, priority=1)
    assert "priority" not in server.body(0)
    baseline = server.body(1)
    assert set(baseline) == {"model", "messages", "max_tokens", "response_format"}


async def test_timeout_per_request_comes_from_settings() -> None:
    server = Server(completion())
    async with client(server, settings(llm_timeout_s=42)) as llm:
        await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=10)
    assert server.requests[0].extensions["timeout"]["read"] == 42


async def test_unknown_alias_is_a_programming_error() -> None:
    async with client(Server(completion())) as llm:
        with pytest.raises(ValueError, match="no request profile for alias 'gpt-4o'"):
            await llm.call("gpt-4o", MESSAGES, SCHEMA, max_tokens=10)


# --- Parsing -----------------------------------------------------------------------------


async def test_short_keys_mapped_to_full_names_and_numbers_are_decimal() -> None:
    content = '{"po_no": "PO/2025-26/00042", "v_gstin": "27AAPFU0939F1ZV", "gr_total": 1234.50}'
    server = Server(completion(content))
    async with client(server) as llm:
        result = await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=10, call_id="H")
    assert result.parsed == {
        "po_number": "PO/2025-26/00042",
        "vendor_gstin": "27AAPFU0939F1ZV",
        "grand_total": Decimal("1234.50"),
    }
    assert result.call_id == "H" and result.alias == "po-fast" and result.attempts == 1
    assert result.raw_content == content and result.reasoning_content is None
    assert result.tokens == Tokens(prompt=1200, completion=30, reasoning=None)
    assert result.timings.end_ts >= result.timings.start_ts and result.timings.duration_ms >= 0


async def test_line_item_rows_are_kept_as_they_are() -> None:
    server = Server(completion('{"rows": [[1, "BLT-1", "Hex bolt", "731815", 100, 12.5]]}'))
    async with client(server) as llm:
        result = await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=10)
    assert result.parsed == {"rows": [[1, "BLT-1", "Hex bolt", "731815", 100, Decimal("12.5")]]}


async def test_reasoning_is_separated_and_reported_tokens_kept() -> None:
    usage = {
        "prompt_tokens": 900,
        "completion_tokens": 400,
        "total_tokens": 1300,
        "completion_tokens_details": {"reasoning_tokens": 380},
    }
    server = Server(
        completion('{"po_no": "A"}', reasoning="Looking at page 1...", usage=usage),
        completion('<think>check the label</think>{"po_no": "B"}'),
    )
    async with client(server) as llm:
        first = await llm.call("po-accurate", MESSAGES, SCHEMA, max_tokens=10)
        second = await llm.call("po-accurate", MESSAGES, SCHEMA, max_tokens=10)
    assert first.parsed == {"po_number": "A"} and first.reasoning_content == "Looking at page 1..."
    assert first.tokens.reasoning == 380
    assert second.parsed == {"po_number": "B"} and second.reasoning_content == "check the label"


async def test_truncated_answer_is_an_error_result() -> None:
    server = Server(completion('{"rows": [[1, "BLT-1", "Hex bo', finish_reason="length"))
    async with client(server) as llm:
        result = await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=6000)
    assert not result.ok and result.error_kind == "truncated" and result.parsed is None
    assert "max_tokens=6000" in result.error
    assert result.raw_content.startswith('{"rows"') and result.finish_reason == "length"
    assert result.attempts == 1  # never retried here: not a transient error


@pytest.mark.parametrize("content", ["", '{"po_no": ', "not json", '["a", "b"]'])
async def test_invalid_json_is_an_error_result(content: str) -> None:
    async with client(Server(completion(content))) as llm:
        result = await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=10)
    assert not result.ok and result.error_kind == "invalid_json" and result.parsed is None


# --- Transient errors --------------------------------------------------------------------


def raise_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def raise_connect(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


@pytest.mark.parametrize("failure", [503, 429, 500, raise_timeout, raise_connect])
async def test_transient_error_is_retried_then_succeeds(failure) -> None:
    server = Server(failure, completion())
    async with client(server) as llm:
        result = await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=10)
    assert result.ok and result.attempts == 2 and len(server.requests) == 2
    assert result.parsed == {"po_number": "PO/1"}


async def test_transient_retries_are_capped() -> None:
    server = Server(503)
    async with client(server, settings(llm_transient_retries=2)) as llm:
        result = await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=10)
    assert not result.ok and result.error_kind == "transient"
    assert result.attempts == 3 and len(server.requests) == 3  # 1 try + 2 retries
    assert "503" in result.error


@pytest.mark.parametrize("status", [400, 401, 404, 422])
async def test_client_errors_are_not_retried(status: int) -> None:
    server = Server(status)
    async with client(server) as llm:
        result = await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=10)
    assert not result.ok and result.error_kind == "api" and len(server.requests) == 1


async def test_backoff_doubles(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("app.extraction.llm_client.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("app.extraction.llm_client.random.random", lambda: 0.0)
    server = Server(503, 503, completion())
    async with client(server, settings(llm_backoff_s=1.5)) as llm:
        result = await llm.call("po-fast", MESSAGES, SCHEMA, max_tokens=10)
    assert result.ok and delays == [1.5, 3.0]
