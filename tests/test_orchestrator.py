"""Tests for the orchestrator (Step 3.5) with a fake client that sleeps. No LLM, no network."""

import asyncio
import time
from typing import Any

import pytest

from app.core.settings import Settings
from app.extraction.llm_client import CallResult, Timings, Tokens
from app.extraction.orchestrator import llm_semaphore, run_calls
from app.extraction.planner import plan_calls
from app.extraction.preprocess import PageContent, PreparedDoc

SETTINGS = Settings(_env_file=None)


def rows_text(first: int, last: int) -> str:
    return "\n".join(f"{n}\nHex bolt\n731815\n100 NOS" for n in range(first, last + 1))


DOC = PreparedDoc(
    "0" * 64,
    4,
    [
        PageContent(n, "native", rows_text(n * 5 - 4, n * 5), "AAAA", "image/jpeg", 10, 14)
        for n in range(1, 5)
    ],
)


class FakeClient:
    """Sleeps `durations[call_id]` seconds per call and records concurrency and arguments."""

    def __init__(self, durations: dict[str, float], default: float = 0.1) -> None:
        self.durations, self.default = durations, default
        self.in_flight = self.max_in_flight = 0
        self.calls: list[dict[str, Any]] = []
        self.fail: dict[str, str] = {}  # call_id -> "error" | "raise"

    async def call(self, alias, messages, schema, max_tokens, priority=None, call_id=None):
        self.calls.append(
            dict(alias=alias, call_id=call_id, priority=priority, schema=schema, messages=messages)
        )
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.durations.get(call_id, self.default))
            if self.fail.get(call_id) == "raise":
                raise RuntimeError("boom")
        finally:
            self.in_flight -= 1
        error = "HTTP 503" if self.fail.get(call_id) == "error" else None
        now = time.time()
        return CallResult(
            call_id=call_id,
            alias=alias,
            parsed=None if error else {"ok": True},
            raw_content="{}",
            reasoning_content=None,
            tokens=Tokens(),
            timings=Timings(now, now, 0.0),
            attempts=1,
            error=error,
            error_kind="transient" if error else None,
        )


def overlaps(a, b) -> bool:
    return a.start_ms < b.end_ms and b.start_ms < a.end_ms


async def test_two_call_runs_both_calls_at_once() -> None:
    plan = plan_calls(DOC, SETTINGS, "two_call")
    client = FakeClient({"H": 0.2, "LI": 0.3})
    start = time.perf_counter()
    outcomes = await run_calls(DOC, plan.specs, 5, client, SETTINGS, semaphore=asyncio.Semaphore(8))
    total = time.perf_counter() - start
    assert 0.3 <= total < 0.45  # about the slowest call, not the sum (0.5 s)
    header, lines = outcomes
    assert overlaps(header.timeline, lines.timeline)
    assert [o.spec.call_id for o in outcomes] == ["H", "LI"]
    assert all(o.result.ok and o.timeline.ok for o in outcomes)


async def test_per_page_calls_all_overlap() -> None:
    plan = plan_calls(DOC, SETTINGS, "per_page")
    assert [s.call_id for s in plan.specs] == ["H", "LI-p1", "LI-p2", "LI-p3", "LI-p4"]
    client = FakeClient({"LI-p3": 0.3}, default=0.15)
    start = time.perf_counter()
    outcomes = await run_calls(DOC, plan.specs, 1, client, SETTINGS, semaphore=asyncio.Semaphore(8))
    assert time.perf_counter() - start < 0.45  # sequential would be 0.9 s
    assert client.max_in_flight == 5
    assert all(o.timeline.start_ms < 50 for o in outcomes)  # all started right away


async def test_semaphore_caps_concurrency() -> None:
    plan = plan_calls(DOC, SETTINGS, "per_page")
    client = FakeClient({}, default=0.1)
    start = time.perf_counter()
    outcomes = await run_calls(DOC, plan.specs, 1, client, SETTINGS, semaphore=asyncio.Semaphore(2))
    total = time.perf_counter() - start
    assert client.max_in_flight == 2
    assert 0.3 <= total < 0.45  # 5 calls, 2 at a time: 3 waves of 0.1 s
    waits = sorted(o.timeline.wait_ms for o in outcomes)
    assert waits[:2] == pytest.approx([0, 0], abs=20) and waits[-1] >= 150


async def test_default_semaphore_is_process_wide_and_uses_settings() -> None:
    s = Settings(_env_file=None, llm_concurrency=3)
    assert llm_semaphore(3) is llm_semaphore(99)  # one per event loop, created once
    client = FakeClient({}, default=0.05)
    plan = plan_calls(DOC, s, "per_page")
    await asyncio.gather(*(run_calls(DOC, plan.specs, i, client, s) for i in range(3)))
    assert client.max_in_flight == 3  # 15 calls of 3 POs share the limit


async def test_priority_schema_and_messages_passed_to_the_client() -> None:
    plan = plan_calls(DOC, SETTINGS, "two_call")
    client = FakeClient({}, default=0.01)
    await run_calls(DOC, plan.specs, 42, client, SETTINGS, semaphore=asyncio.Semaphore(8))
    by_id = {c["call_id"]: c for c in client.calls}
    assert {c["priority"] for c in client.calls} == {42}
    assert "po_no" in by_id["H"]["schema"]["properties"]
    assert "rows" in by_id["LI"]["schema"]["properties"]
    assert by_id["H"]["messages"][0] == by_id["LI"]["messages"][0]  # same system prompt


async def test_timeline_is_relative_to_po_start() -> None:
    po_start = time.perf_counter()
    await asyncio.sleep(0.1)  # e.g. pre-processing before the calls
    plan = plan_calls(DOC, SETTINGS, "two_call")
    client = FakeClient({}, default=0.05)
    semaphore = asyncio.Semaphore(8)
    outcomes = await run_calls(DOC, plan.specs, 1, client, SETTINGS, po_start, semaphore)
    for o in outcomes:
        assert 100 <= o.timeline.start_ms < o.timeline.end_ms
        assert o.timeline.end_ms - o.timeline.start_ms >= 50


async def test_one_failed_call_never_cancels_the_others() -> None:
    plan = plan_calls(DOC, SETTINGS, "per_page")
    client = FakeClient({"LI-p2": 0.3}, default=0.05)
    client.fail = {"H": "error", "LI-p1": "raise"}
    outcomes = await run_calls(DOC, plan.specs, 1, client, SETTINGS, semaphore=asyncio.Semaphore(8))
    by_id = {o.spec.call_id: o for o in outcomes}
    assert len(outcomes) == 5
    assert by_id["H"].result.error_kind == "transient" and not by_id["H"].timeline.ok
    assert by_id["LI-p1"].result.error_kind == "internal"
    assert "RuntimeError: boom" in by_id["LI-p1"].result.error
    assert by_id["LI-p2"].result.ok  # the slow call still finished
    assert by_id["LI-p3"].result.ok and by_id["LI-p4"].result.ok
