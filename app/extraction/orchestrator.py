"""Orchestrator: run the planned LLM calls of one PO concurrently (PRD FR-14, FR-15).

- All calls of a PO start together (asyncio.gather); with a shared page prefix, the server
  reads the pages once and the PO takes about as long as its slowest call.
- A process-wide semaphore caps in-flight LLM calls at settings.llm_concurrency. Keep it at
  or below vLLM --max-num-seqs (more would only queue inside vLLM, out of our sight) and never
  set LiteLLM limits lower than it.
- Every call carries the PO's priority: its arrival sequence number. Smaller = earlier = served
  first by vLLM priority scheduling, so a burst finishes POs in order instead of all late.
- Each call gets a timeline entry relative to the PO start, for the dashboard Gantt view:
  wait (queued for the semaphore), then start..end (the LLM call, retries included).
- A failed call never cancels the others. The client returns errors as results; anything
  unexpected is turned into a failed result too. Step 3.7 decides what to retry.
"""

import asyncio
import time
import weakref
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.logging import get_logger
from app.core.settings import Settings
from app.extraction.llm_client import CallResult, Timings, Tokens
from app.extraction.planner import CallSpec
from app.extraction.preprocess import PreparedDoc
from app.extraction.prompts import build_messages
from schema.llm_schemas import header_schema_for_llm, line_items_schema_for_llm

log = get_logger(__name__)


class LlmCaller(Protocol):
    """What the orchestrator needs from a client (LlmClient, or a fake in tests)."""

    async def call(
        self,
        alias: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        max_tokens: int,
        priority: int | None = None,
        call_id: str | None = None,
    ) -> CallResult: ...


@dataclass(frozen=True)
class TimelineEntry:
    """When one call ran, in ms since the PO started."""

    call_id: str
    alias: str
    wait_ms: float  # queued for the concurrency limit
    start_ms: float  # request sent
    end_ms: float  # result received
    ok: bool


@dataclass(frozen=True)
class CallOutcome:
    """A planned call, its result and its timeline entry."""

    spec: CallSpec
    result: CallResult
    timeline: TimelineEntry


_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def llm_semaphore(limit: int) -> asyncio.Semaphore:
    """The process-wide limit on in-flight LLM calls (one per event loop; a worker has one).

    Created on first use with `limit`; later calls return the same semaphore.
    """
    loop = asyncio.get_running_loop()
    if loop not in _SEMAPHORES:
        _SEMAPHORES[loop] = asyncio.Semaphore(limit)
    return _SEMAPHORES[loop]


def _schema(spec: CallSpec) -> dict[str, Any]:
    return header_schema_for_llm() if spec.task.kind == "header" else line_items_schema_for_llm()


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def _run_one(
    doc: PreparedDoc,
    spec: CallSpec,
    po_priority: int | None,
    client: LlmCaller,
    semaphore: asyncio.Semaphore,
    po_start: float,
) -> CallOutcome:
    queued = time.perf_counter()
    async with semaphore:
        wait_ms = _ms_since(queued)
        start_ms = _ms_since(po_start)
        try:
            result = await client.call(
                spec.alias,
                build_messages(doc, spec.task),
                _schema(spec),
                spec.max_tokens,
                priority=po_priority,
                call_id=spec.call_id,
            )
        except Exception as exc:  # a bug, not an API error: report it, keep the other calls
            log.exception("LLM call raised", extra={"call_id": spec.call_id, "alias": spec.alias})
            now = time.time()
            result = CallResult(
                call_id=spec.call_id,
                alias=spec.alias,
                parsed=None,
                raw_content=None,
                reasoning_content=None,
                tokens=Tokens(),
                timings=Timings(now, now, 0.0),
                attempts=1,
                error=f"{type(exc).__name__}: {exc}",
                error_kind="internal",
            )
        end_ms = _ms_since(po_start)
    timeline = TimelineEntry(spec.call_id, spec.alias, wait_ms, start_ms, end_ms, result.ok)
    return CallOutcome(spec, result, timeline)


async def run_calls(
    doc: PreparedDoc,
    specs: list[CallSpec],
    po_priority: int | None,
    client: LlmCaller,
    settings: Settings,
    po_start: float | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> list[CallOutcome]:
    """Run all `specs` of one PO concurrently; outcomes come back in spec order.

    `po_start` is the PO's start (time.perf_counter()) for the timeline; default: now.
    `semaphore` defaults to the process-wide llm_semaphore(settings.llm_concurrency).
    """
    po_start = time.perf_counter() if po_start is None else po_start
    semaphore = semaphore or llm_semaphore(settings.llm_concurrency)
    return list(
        await asyncio.gather(
            *(_run_one(doc, spec, po_priority, client, semaphore, po_start) for spec in specs)
        )
    )
