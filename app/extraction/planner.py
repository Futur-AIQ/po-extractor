"""Call planner: which LLM calls extract one PO (PRD FR-07, §9.3).

    two_call   H + LI           header and all line items; both calls see every page
    per_page   H + LI-p{n}      header + one line-item call per item page
    adaptive   two_call, but line items per page when the PO is long (default)

All calls share the page prefix (prompts.py), so extra calls re-read the pages from the
prefix cache. Per-page calls cap the slowest call on long POs at the cost of more requests.

Item pages and row counts come from the text layer of native pages. The text engine gives
plain text, where each row starts with its serial number on a line of its own or followed by
a space ("17" or "17 BRG/7080/A ..."). The row estimate is the longest chain of consecutive
serial numbers (n, n+1, ...) in reading order, so numbers such as "18 %" do not count, and
numbered clauses ("1. Payment: ...", a dot after the number) are not rows. On the synthetic
set this finds every page with rows and no other page, and is exact or one too high.

Pages without a text layer (scanned, or native in native_mode "image") always count as item
pages: their rows are unknown. If no native page shows a row, every page counts, because a
missed item page would lose rows while an extra page only costs a call that returns [].
"""

import re
from dataclasses import dataclass
from typing import Literal

from app.core.settings import Settings
from app.extraction.preprocess import PreparedDoc
from app.extraction.prompts import Task, header_task, lines_task

Strategy = Literal["two_call", "per_page", "adaptive"]

_SERIAL = re.compile(r"^[ \t]*([0-9]{1,3})(?:[ \t]|$)", re.M)


@dataclass(frozen=True)
class CallSpec:
    """One planned LLM call."""

    call_id: str  # "H", "LI" or "LI-p{n}"
    task: Task
    alias: str
    max_tokens: int


@dataclass(frozen=True)
class CallPlan:
    """The calls for one PO and why (routing.strategy / routing.reason in the result)."""

    specs: list[CallSpec]
    requested: Strategy
    strategy: Literal["two_call", "per_page"]
    reason: str
    estimated_rows: int  # from native text layers only
    item_pages: list[int]


def estimate_rows(text: str) -> int:
    """Number of line-item rows that start in `text`: the longest serial-number chain."""
    chain: dict[int, int] = {}  # serial number -> length of the chain ending at it
    for match in _SERIAL.finditer(text):
        number = int(match.group(1))
        if number > 0:
            chain[number] = max(chain.get(number, 0), chain.get(number - 1, 0) + 1)
    return max(chain.values(), default=0)


def rows_by_page(doc: PreparedDoc) -> dict[int, int]:
    """Estimated rows per kept page that has a text layer."""
    return {p.page_no: estimate_rows(p.text_md) for p in doc.kept_pages if p.text_md is not None}


def item_pages(doc: PreparedDoc) -> list[int]:
    """Pages to send a per-page line-item call for (see module docstring)."""
    rows = rows_by_page(doc)
    if not any(rows.values()):  # no row found in any text layer: send every page
        return [p.page_no for p in doc.kept_pages]
    return [p.page_no for p in doc.kept_pages if rows.get(p.page_no, 1) > 0]


def plan_calls(doc: PreparedDoc, settings: Settings, strategy: Strategy | None = None) -> CallPlan:
    """Plan the calls for one PO with `strategy` (default: settings.call_strategy)."""
    requested = strategy or settings.call_strategy
    pages = item_pages(doc)
    rows_per_page = rows_by_page(doc)
    rows = sum(rows_per_page.values())
    no_text = len(doc.kept_pages) - len(rows_per_page)

    if requested == "adaptive":
        row_limit, page_limit = settings.adaptive_row_threshold, settings.adaptive_page_threshold
        long_po = rows > row_limit, len(pages) > page_limit
        chosen = "per_page" if any(long_po) else "two_call"
        row_part = (
            f"{rows} estimated rows {'>' if long_po[0] else '<='} {row_limit}"
            if rows_per_page
            else "rows unknown (no text layer)"
        )
        page_part = f"{len(pages)} item pages {'>' if long_po[1] else '<='} {page_limit}"
        if no_text and rows_per_page:
            page_part += f" ({no_text} without text, rows unknown)"
        reason = f"adaptive: {row_part}, {page_part} -> {chosen}"
    else:
        chosen = requested
        reason = f"{requested} requested"

    header = CallSpec("H", header_task(), settings.po_fast, settings.llm_max_tokens_header)
    if chosen == "two_call":
        lines = [CallSpec("LI", lines_task(), settings.po_fast, settings.llm_max_tokens_lines)]
    else:
        lines = [
            CallSpec(f"LI-p{n}", lines_task(n), settings.po_fast, settings.llm_max_tokens_lines)
            for n in pages
        ]
    return CallPlan([header, *lines], requested, chosen, reason, rows, pages)
