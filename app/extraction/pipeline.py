"""End-to-end extraction of one PO: extract(pdf_bytes) -> result envelope (PRD §6.4).

    prepare -> plan -> calls on po-fast -> merge -> compute -> validate
      all rules pass              -> completed
      a non-retryable failure     -> needs_review, no retry (e.g. a duplicate PO)
      retryable failures          -> re-run ONLY the failed calls on po-accurate (thinking on),
                                     then merge, compute, validate again
                                       pass -> completed (routing.retried_calls)
                                       fail -> needs_review
      a call error after the client's transient retries -> failed (the job layer's DLQ)

Retries happen at most settings.max_reasks times (PRD FR-11: once) and only while the PO has
run for less than settings.retry_deadline_s, so a retry never pushes a PO past the 60 s
ceiling. needs_review results list the failed rules and fields for the dashboard.

A truncated or invalid-JSON answer is not a call error: the model answered, badly. It shows up
as a failed schema check and its call is retried like any other failed call.
"""

import hashlib
import time
import uuid
from dataclasses import replace
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from app.core.settings import Settings
from app.extraction.compute import compute
from app.extraction.grounding import GroundingIndex
from app.extraction.merge import MergedPO, merge_outcomes
from app.extraction.orchestrator import CallOutcome, LlmCaller, run_calls
from app.extraction.planner import CallPlan, CallSpec, plan_calls
from app.extraction.preprocess import PreparedDoc, prepare
from app.extraction.validate import (
    Masters,
    ValidationReport,
    load_masters,
    to_purchase_order,
    validate,
)

Status = Literal["completed", "needs_review", "failed"]
CALL_ERRORS = {"transient", "api", "internal"}  # the request failed, not the answer


# =========================================================================================
# Result envelope (PRD §6.4)
# =========================================================================================


class CheckOut(BaseModel):
    rule: str
    passed: bool
    fields: list[str] = Field(default_factory=list)
    detail: str = ""
    retry_target: str | None = None
    retryable: bool = True


class ValidationOut(BaseModel):
    passed: bool
    checks: list[CheckOut]


class RoutingOut(BaseModel):
    strategy: str  # two_call | per_page
    requested_strategy: str  # adds adaptive
    reason: str
    estimated_rows: int
    item_pages: list[int]
    retried_calls: list[str] = Field(default_factory=list)
    retry_alias: str | None = None
    retry_skipped: str | None = None  # why a retry did not happen


class PageOut(BaseModel):
    page: int
    kind: str
    skipped: bool
    skip_reason: str | None = None


class TimelineOut(BaseModel):
    """One call on the PO's timeline, in ms since the PO started (dashboard Gantt view)."""

    call_id: str
    alias: str
    attempt: Literal["first", "retry"]
    wait_ms: float
    start_ms: float
    end_ms: float
    ok: bool
    error: str | None = None


class ReviewOut(BaseModel):
    failed_rules: list[str]
    failed_fields: list[str]


class ResultEnvelope(BaseModel):
    """The result of one PO, as stored and returned by the API."""

    job_id: str
    file_sha256: str
    status: Status
    error: str | None = None  # status failed: why
    data: dict[str, Any] | None = None  # the PO (partial when it needs review)
    validation: ValidationOut | None = None
    review: ReviewOut | None = None  # needs_review: what to look at
    routing: RoutingOut | None = None
    pages: list[PageOut] = Field(default_factory=list)
    timings_ms: dict[str, Any] = Field(default_factory=dict)
    tokens: dict[str, dict[str, int | None]] = Field(default_factory=dict)
    timeline: list[TimelineOut] = Field(default_factory=list)


# =========================================================================================
# Helpers
# =========================================================================================


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def default_masters(settings: Settings) -> Masters | None:
    """Master data from settings.masters_dir, or None if it is not there."""
    path = settings.masters_dir
    return load_masters(path) if (path / "parties.json").exists() else None


def calls_to_retry(report: ValidationReport, plan: CallPlan) -> list[str]:
    """Call ids to re-run for the failed checks' retry targets, in plan order."""
    line_calls = [spec.call_id for spec in plan.specs if spec.task.kind == "lines"]
    wanted: set[str] = set()
    for check in report.failed:
        target = check.retry_target
        if target in ("H", "both"):
            wanted.add("H")
        if target in ("LI", "both"):
            wanted.update(line_calls)
        elif target is not None and target.startswith("LI"):
            wanted.add(target)
    return [spec.call_id for spec in plan.specs if spec.call_id in wanted]


def _call_errors(outcomes: list[CallOutcome]) -> list[str]:
    return [
        f"{o.spec.call_id}: {o.result.error}"
        for o in outcomes
        if o.result.error_kind in CALL_ERRORS
    ]


def _data(merged: MergedPO) -> dict[str, Any]:
    """The PO as JSON-ready data: complete if possible, else what was extracted."""
    po = to_purchase_order(merged)
    if po is not None:
        return po.model_dump(mode="json")
    partial = {**merged.header, "line_items": [item.model_dump() for item in merged.items]}
    return to_jsonable_python(partial)


def _timeline(outcomes: list[CallOutcome], attempt: str) -> list[TimelineOut]:
    return [
        TimelineOut(
            call_id=o.spec.call_id,
            alias=o.spec.alias,
            attempt=attempt,
            wait_ms=o.timeline.wait_ms,
            start_ms=o.timeline.start_ms,
            end_ms=o.timeline.end_ms,
            ok=o.result.ok,
            error=o.result.error,
        )
        for o in outcomes
    ]


def _tokens(outcomes: list[CallOutcome], suffix: str = "") -> dict[str, dict[str, int | None]]:
    return {
        o.spec.call_id + suffix: {
            "in": o.result.tokens.prompt,
            "out": o.result.tokens.completion,
            "reasoning": o.result.tokens.reasoning,
        }
        for o in outcomes
    }


def _validation(report: ValidationReport) -> ValidationOut:
    checks = [CheckOut(**{**vars(c), "fields": list(c.fields)}) for c in report.checks]
    return ValidationOut(passed=report.passed, checks=checks)


def _review(report: ValidationReport) -> ReviewOut:
    fields = [f for check in report.failed for f in check.fields]
    return ReviewOut(
        failed_rules=[check.rule for check in report.failed],
        failed_fields=list(dict.fromkeys(fields)),
    )


# =========================================================================================
# One PO
# =========================================================================================


async def extract(
    pdf_bytes: bytes,
    client: LlmCaller,
    settings: Settings,
    *,
    job_id: str | None = None,
    po_priority: int | None = None,
    masters: Masters | None = None,
    queue_ms: float = 0.0,
) -> ResultEnvelope:
    """Extract one PO end to end (see module docstring). Never raises for a bad PDF or a
    failed call: those give status "failed".

    `po_priority`: the PO's arrival sequence number (smaller = served first by vLLM).
    `masters`: ERP master data; default: settings.masters_dir if present.
    `queue_ms`: time the job waited in the queue, for the timings.
    """
    start = time.perf_counter()
    job_id = job_id or uuid.uuid4().hex
    masters = masters if masters is not None else default_masters(settings)
    timings: dict[str, Any] = {"queue": queue_ms}

    try:
        doc: PreparedDoc = await prepare(pdf_bytes, settings)
    except ValueError as exc:  # not a readable PDF
        sha = hashlib.sha256(pdf_bytes).hexdigest()
        timings["total"] = _ms(start)
        return ResultEnvelope(
            job_id=job_id, file_sha256=sha, status="failed", error=str(exc), timings_ms=timings
        )
    timings["preprocess"] = _ms(start)
    pages = [
        PageOut(page=p.page_no, kind=p.kind, skipped=p.skipped, skip_reason=p.skip_reason)
        for p in doc.pages
    ]

    plan = plan_calls(doc, settings)
    routing = RoutingOut(
        strategy=plan.strategy,
        requested_strategy=plan.requested,
        reason=plan.reason,
        estimated_rows=plan.estimated_rows,
        item_pages=plan.item_pages,
    )

    llm_start = time.perf_counter()
    outcomes = await run_calls(doc, plan.specs, po_priority, client, settings, po_start=start)
    timings["llm"] = _ms(llm_start)
    timings["llm_calls"] = {o.spec.call_id: o.result.timings.duration_ms for o in outcomes}
    timeline = _timeline(outcomes, "first")
    tokens = _tokens(outcomes)

    def envelope(status: Status, **fields: Any) -> ResultEnvelope:
        timings["total"] = _ms(start)
        return ResultEnvelope(
            job_id=job_id,
            file_sha256=doc.doc_sha256,
            status=status,
            routing=routing,
            pages=pages,
            timings_ms=timings,
            tokens=tokens,
            timeline=timeline,
            **fields,
        )

    if errors := _call_errors(outcomes):
        return envelope("failed", error="; ".join(errors))

    validate_start = time.perf_counter()
    grounding = GroundingIndex.from_doc(doc)
    merged = compute(merge_outcomes(outcomes))
    report = validate(merged, grounding, masters)
    timings["validate"] = _ms(validate_start)

    if not report.passed and settings.max_reasks == 0:
        routing.retry_skipped = "retries disabled (MAX_REASKS=0)"
    for _ in range(settings.max_reasks):
        if report.passed:
            break
        if any(not check.retryable for check in report.failed):
            rules = [c.rule for c in report.failed if not c.retryable]
            routing.retry_skipped = f"not retryable: {', '.join(rules)}"
            break
        elapsed_s = time.perf_counter() - start
        if elapsed_s > settings.retry_deadline_s:
            routing.retry_skipped = (
                f"retry deadline: {elapsed_s:.1f} s elapsed > {settings.retry_deadline_s:g} s"
            )
            break
        retry_ids = calls_to_retry(report, plan)
        if not retry_ids:
            routing.retry_skipped = "no call to retry"
            break

        retry_start = time.perf_counter()
        extra = settings.llm_retry_reasoning_tokens
        specs: list[CallSpec] = [
            replace(spec, alias=settings.po_accurate, max_tokens=spec.max_tokens + extra)
            for spec in plan.specs
            if spec.call_id in retry_ids
        ]
        retried = await run_calls(doc, specs, po_priority, client, settings, po_start=start)
        timeline += _timeline(retried, "retry")
        tokens |= _tokens(retried, ":retry")
        routing.retried_calls += retry_ids
        routing.retry_alias = settings.po_accurate
        # A retry that itself failed keeps the first answer for that call.
        better = {o.spec.call_id: o for o in retried if o.result.ok}
        outcomes = [better.get(o.spec.call_id, o) for o in outcomes]
        merged = compute(merge_outcomes(outcomes))
        report = validate(merged, grounding, masters)
        timings["retry"] = timings.get("retry", 0) + _ms(retry_start)

    validation = _validation(report)
    if report.passed:
        return envelope("completed", data=_data(merged), validation=validation)
    review = _review(report)
    return envelope("needs_review", data=_data(merged), validation=validation, review=review)
