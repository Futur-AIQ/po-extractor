"""Tests for the end-to-end pipeline (Step 3.7) with a fake LLM client. No network.

Self-contained: a generated PO (datagen), a small two-page text PDF with its identifiers,
and a fake client that answers with the PO's own values, optionally tampered with.
"""

import time
from collections.abc import Callable
from typing import Any

import pymupdf
import pytest

from app.core.settings import Settings
from app.extraction.grounding import normalise
from app.extraction.llm_client import CallResult, Timings, Tokens
from app.extraction.pipeline import ResultEnvelope, extract
from app.extraction.validate import Masters
from datagen.generate import Knobs, generate_po
from schema.fields import COMPUTED_FIELDS, LLM_LINE_ITEM_COLUMNS
from schema.po_schema import PurchaseOrder

PO: PurchaseOrder = generate_po(
    11, Knobs(min_lines=7, max_lines=7, typical_min_lines=7, typical_max_lines=7)
).po
PAGE_ROWS = {1: [1, 2, 3], 2: [4, 5, 6, 7]}


def make_pdf(po: PurchaseOrder) -> bytes:
    """Two native pages: header identifiers and rows 1-3, then rows 4-7."""
    head = [f"PURCHASE ORDER {po.po_number}", f"Buyer {po.buyer_name}"]
    head += [
        f"{label} {value}"
        for label, value in (
            ("Buyer GSTIN", po.buyer_gstin),
            ("Buyer PAN", po.buyer_pan),
            ("Vendor GSTIN", po.vendor_gstin),
            ("Vendor PAN", po.vendor_pan),
            ("Bill to GSTIN", po.bill_to_gstin),
            ("Ship to GSTIN", po.ship_to_gstin),
        )
        if value
    ]
    with pymupdf.open() as doc:
        for page_no, numbers in PAGE_ROWS.items():
            lines = head if page_no == 1 else [f"Page {page_no}"]
            for item in po.line_items:
                if item.line_no in numbers:
                    lines += [str(item.line_no), item.item_code or "", "Item", item.hsn_sac]
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), "\n".join(lines), fontsize=8)
        return doc.tobytes()


PDF = make_pdf(PO)


def masters(processed: set[str] = frozenset()) -> Masters:
    parties = {
        g for g in (PO.buyer_gstin, PO.vendor_gstin, PO.bill_to_gstin, PO.ship_to_gstin) if g
    }
    codes = frozenset(item.item_code for item in PO.line_items if item.item_code)
    pan = PO.buyer_gstin[2:12]
    return Masters(
        frozenset(parties), {pan: codes}, {pan: frozenset(normalise(n) for n in processed)}
    )


def header_answer() -> dict[str, Any]:
    data = PO.model_dump(mode="json", exclude={"line_items", *COMPUTED_FIELDS}, exclude_none=True)
    return data


def rows_answer(numbers: list[int] | None = None) -> dict[str, Any]:
    lines = PO.model_dump(mode="json")["line_items"]
    return {
        "rows": [
            [line[c] for c in LLM_LINE_ITEM_COLUMNS]
            for line in lines
            if numbers is None or line["line_no"] in numbers
        ]
    }


Tamper = Callable[[dict[str, Any]], dict[str, Any] | str]  # an answer, or an error kind


class FakeClient:
    """Answers each call with the PO's own values; `tamper[(alias, call_id)]` changes one."""

    def __init__(self, tamper: dict[tuple[str, str], Tamper] | None = None) -> None:
        self.tamper = tamper or {}
        self.calls: list[tuple[str, str, int, int | None]] = []  # alias, call_id, max_tokens, prio

    async def call(self, alias, messages, schema, max_tokens, priority=None, call_id=None):
        self.calls.append((alias, call_id, max_tokens, priority))
        if call_id == "H":
            parsed: dict[str, Any] | str = header_answer()
        elif call_id == "LI":
            parsed = rows_answer()
        else:
            parsed = rows_answer(PAGE_ROWS[int(call_id.removeprefix("LI-p"))])
        if (alias, call_id) in self.tamper:
            parsed = self.tamper[(alias, call_id)](parsed)
        now = time.time()
        kind = parsed if isinstance(parsed, str) else None
        return CallResult(
            call_id=call_id,
            alias=alias,
            parsed=None if kind else parsed,
            raw_content="{}",
            reasoning_content=None,
            tokens=Tokens(prompt=1000, completion=50),
            timings=Timings(now, now, 5.0),
            attempts=1,
            error=f"fake {kind}" if kind else None,
            error_kind=kind,
        )

    def aliases(self, call_id: str) -> list[str]:
        return [alias for alias, cid, _, _ in self.calls if cid == call_id]


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **{"call_strategy": "two_call", **overrides})


async def run(client: FakeClient, s: Settings | None = None, **kw) -> ResultEnvelope:
    return await extract(PDF, client, s or settings(), masters=kw.pop("m", masters()), **kw)


def wrong_gstin(answer: dict[str, Any]) -> dict[str, Any]:
    gstin = answer["vendor_gstin"]
    return {**answer, "vendor_gstin": gstin[:14] + ("A" if gstin[14] != "A" else "B")}


def drop_line(n: int) -> Tamper:
    return lambda answer: {"rows": [r for r in answer["rows"] if r[0] != n]}


# --- Happy path --------------------------------------------------------------------------


async def test_clean_po_completes_without_retry() -> None:
    client = FakeClient()
    env = await run(client, po_priority=7, job_id="job-1")
    assert env.status == "completed" and env.error is None and env.review is None
    assert env.job_id == "job-1" and len(env.file_sha256) == 64
    assert env.validation.passed and len(env.validation.checks) == 11
    assert env.data["po_number"] == PO.po_number
    assert env.data["grand_total"] == str(PO.grand_total)
    assert len(env.data["line_items"]) == 7 and env.data["total_tax"] is not None
    assert env.routing.strategy == "two_call" and env.routing.retried_calls == []
    assert [p.kind for p in env.pages] == ["native", "native"]
    assert [(t.call_id, t.attempt) for t in env.timeline] == [("H", "first"), ("LI", "first")]
    assert set(env.tokens) == {"H", "LI"} and env.tokens["H"]["in"] == 1000
    assert {"queue", "preprocess", "llm", "llm_calls", "validate", "total"} <= set(env.timings_ms)
    assert [(a, c, p) for a, c, _, p in client.calls] == [("po-fast", "H", 7), ("po-fast", "LI", 7)]
    env.model_dump_json()  # JSON-serialisable for the API and storage


# --- Targeted retries --------------------------------------------------------------------


async def test_wrong_gstin_retries_only_the_header_on_po_accurate() -> None:
    client = FakeClient({("po-fast", "H"): wrong_gstin})
    env = await run(client)
    assert env.status == "completed"
    assert client.aliases("H") == ["po-fast", "po-accurate"]
    assert client.aliases("LI") == ["po-fast"]  # line items not re-run
    assert env.routing.retried_calls == ["H"] and env.routing.retry_alias == "po-accurate"
    assert [(t.call_id, t.attempt) for t in env.timeline][-1] == ("H", "retry")
    assert "H:retry" in env.tokens and "retry" in env.timings_ms
    retry_tokens = next(m for a, c, m, _ in client.calls if a == "po-accurate")
    assert retry_tokens == 2000 + 4000  # header cap + reasoning allowance


async def test_dropped_row_on_page_2_retries_only_that_page() -> None:
    client = FakeClient({("po-fast", "LI-p2"): drop_line(5)})  # a row in the middle of page 2
    env = await run(client, settings(call_strategy="per_page"))
    assert env.routing.strategy == "per_page"
    assert env.status == "completed"
    assert env.routing.retried_calls == ["LI-p2"]
    assert client.aliases("LI-p2") == ["po-fast", "po-accurate"]
    assert client.aliases("H") == client.aliases("LI-p1") == ["po-fast"]


async def test_truncated_answer_is_retried() -> None:
    client = FakeClient({("po-fast", "LI"): lambda _: "truncated"})
    env = await run(client)
    assert env.status == "completed" and env.routing.retried_calls == ["LI"]


async def test_still_failing_after_retry_needs_review_with_fields() -> None:
    client = FakeClient({("po-fast", "H"): wrong_gstin, ("po-accurate", "H"): wrong_gstin})
    env = await run(client)
    assert env.status == "needs_review" and env.routing.retried_calls == ["H"]
    assert "gstin" in env.review.failed_rules and "vendor_gstin" in env.review.failed_fields
    assert env.data["vendor_gstin"] != PO.vendor_gstin  # the extracted (wrong) value is shown


# --- No retry ----------------------------------------------------------------------------


async def test_duplicate_po_goes_to_review_without_retry() -> None:
    client = FakeClient()
    env = await run(client, m=masters({PO.po_number}))
    assert env.status == "needs_review"
    assert env.review.failed_rules == ["duplicate_po"] and env.review.failed_fields == ["po_number"]
    assert (
        env.routing.retried_calls == []
        and env.routing.retry_skipped == "not retryable: duplicate_po"
    )
    assert all(alias == "po-fast" for alias, *_ in client.calls)


async def test_retry_deadline_exceeded_skips_the_retry() -> None:
    client = FakeClient({("po-fast", "H"): wrong_gstin})
    env = await run(client, settings(retry_deadline_s=0))
    assert env.status == "needs_review" and env.routing.retried_calls == []
    assert env.routing.retry_skipped.startswith("retry deadline:")
    assert "vendor_gstin" in env.review.failed_fields
    assert all(alias == "po-fast" for alias, *_ in client.calls)


async def test_retries_can_be_disabled() -> None:
    client = FakeClient({("po-fast", "H"): wrong_gstin})
    env = await run(client, settings(max_reasks=0))
    assert env.status == "needs_review" and env.routing.retry_skipped.startswith("retries disabled")


# --- Failed ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["transient", "api", "internal"])
async def test_call_error_fails_the_po_without_validation_retry(kind: str) -> None:
    client = FakeClient({("po-fast", "LI"): lambda _: kind})
    env = await run(client)
    assert env.status == "failed" and "LI: fake" in env.error
    assert env.validation is None and env.timeline and env.routing is not None
    assert all(alias == "po-fast" for alias, *_ in client.calls)


async def test_not_a_pdf_fails() -> None:
    env = await extract(b"not a pdf", FakeClient(), settings(), masters=masters())
    assert env.status == "failed" and "not a readable PDF" in env.error and env.timeline == []
