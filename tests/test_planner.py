"""Tests for the call planner (Step 3.5). No LLM, no network.

The dataset tests read data/synthetic (built by `make dataset`) and are skipped without it.
"""

import csv
import json
from pathlib import Path

import pytest

from app.core.settings import Settings
from app.extraction.planner import estimate_rows, item_pages, plan_calls
from app.extraction.preprocess import PageContent, PreparedDoc, prepare_sync

DATASET = Path("data/synthetic")
needs_dataset = pytest.mark.skipif(
    not (DATASET / "manifest.csv").exists(), reason="data/synthetic not built (make dataset)"
)


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def rows_text(first: int, last: int) -> str:
    """A plain-text item table as PyMuPDF prints it: one cell per line."""
    cells = []
    for n in range(first, last + 1):
        cells += [str(n), f"BLT/{n:04d}/A", "Hex bolt M12", "731815", "100 NOS", "18 %", "1,180.00"]
    return "Sl\nNo.\nDescription\n" + "\n".join(cells)


def page(page_no: int, kind: str = "native", text: str | None = "", **kw) -> PageContent:
    text = None if kind == "scanned" else text
    return PageContent(page_no, kind, text, "AAAA", "image/jpeg", 10, 14, **kw)


def doc(*pages: PageContent) -> PreparedDoc:
    return PreparedDoc("0" * 64, len(pages), list(pages))


TERMS = "TERMS AND CONDITIONS\n1. Payment: 30 days\n2. Freight: paid\n3. Validity: 30 days\n"


# --- Row estimate ------------------------------------------------------------------------


def test_estimate_rows_counts_the_serial_number_chain() -> None:
    assert estimate_rows(rows_text(1, 16)) == 16
    assert estimate_rows(rows_text(17, 40)) == 24  # a later page starts mid-sequence
    assert estimate_rows("") == 0


def test_estimate_rows_ignores_noise_clauses_and_codes() -> None:
    noise = "18 %\n29\nCode : 29\n271019\n2,09,914.50\n"
    assert estimate_rows(noise + rows_text(1, 5) + "\n" + noise) == 5
    assert estimate_rows(TERMS) == 0  # numbered clauses have a dot after the number
    assert estimate_rows("1 Hex bolt 100 NOS\n2 Washer 200 NOS\n3 Nut 50 NOS") == 3  # inline rows


# --- Strategies --------------------------------------------------------------------------


def test_two_call_plan() -> None:
    plan = plan_calls(doc(page(1, text=rows_text(1, 3))), settings(), "two_call")
    assert [s.call_id for s in plan.specs] == ["H", "LI"]
    assert (plan.requested, plan.strategy, plan.reason) == (
        "two_call",
        "two_call",
        "two_call requested",
    )
    header, lines = plan.specs
    assert (
        header.task.kind == "header" and lines.task.kind == "lines" and lines.task.page_no is None
    )


def test_per_page_plan_covers_item_pages_only() -> None:
    d = doc(
        page(1, text="PURCHASE ORDER\n" + rows_text(1, 10)),
        page(2, kind="scanned"),  # rows unknown: always an item page
        page(3, text="Sub Total\n1,23,456.00\nGrand Total\n1,45,678.08"),  # totals only
        page(4, text=None, skipped=True, skip_reason="terms only"),
        page(5, text=rows_text(11, 12)),
    )
    plan = plan_calls(d, settings(), "per_page")
    assert [s.call_id for s in plan.specs] == ["H", "LI-p1", "LI-p2", "LI-p5"]
    assert [s.task.page_no for s in plan.specs[1:]] == [1, 2, 5]
    assert plan.item_pages == [1, 2, 5] and plan.estimated_rows == 12


def test_aliases_and_token_caps_come_from_settings() -> None:
    s = settings(po_fast="my-fast", llm_max_tokens_header=111, llm_max_tokens_lines=222)
    plan = plan_calls(doc(page(1, text=rows_text(1, 3))), s, "per_page")
    assert [(c.alias, c.max_tokens) for c in plan.specs] == [("my-fast", 111), ("my-fast", 222)]


def test_no_rows_found_anywhere_sends_every_page() -> None:
    d = doc(page(1, text="PURCHASE ORDER (rows printed as 1. 2. 3.)"), page(2, text=TERMS))
    assert item_pages(d) == [1, 2]


# --- Adaptive ----------------------------------------------------------------------------


@pytest.mark.parametrize(("rows", "expected"), [(40, "two_call"), (41, "per_page")])
def test_adaptive_switches_above_the_row_threshold(rows: int, expected: str) -> None:
    d = doc(page(1, text=rows_text(1, 20)), page(2, text=rows_text(21, rows)))
    plan = plan_calls(d, settings())
    assert (plan.requested, plan.strategy, plan.estimated_rows) == ("adaptive", expected, rows)
    sign = ">" if expected == "per_page" else "<="
    assert f"{rows} estimated rows {sign} 40" in plan.reason and plan.reason.endswith(expected)


@pytest.mark.parametrize(("pages", "expected"), [(2, "two_call"), (3, "per_page")])
def test_adaptive_switches_above_the_page_threshold(pages: int, expected: str) -> None:
    d = doc(*(page(n, text=rows_text(n * 3 - 2, n * 3)) for n in range(1, pages + 1)))
    plan = plan_calls(d, settings())
    assert plan.strategy == expected
    sign = ">" if expected == "per_page" else "<="
    assert f"{pages} item pages {sign} 2" in plan.reason
    expected_ids = (
        ["H", "LI"] if expected == "two_call" else ["H"] + [f"LI-p{n}" for n in range(1, pages + 1)]
    )
    assert [s.call_id for s in plan.specs] == expected_ids


def test_adaptive_thresholds_come_from_settings() -> None:
    d = doc(page(1, text=rows_text(1, 10)))
    assert plan_calls(d, settings(adaptive_row_threshold=9)).strategy == "per_page"
    assert plan_calls(d, settings(adaptive_row_threshold=10)).strategy == "two_call"
    assert plan_calls(d, settings(call_strategy="per_page")).reason == "per_page requested"


def test_adaptive_on_scanned_pages_uses_the_page_count_only() -> None:
    plan = plan_calls(doc(page(1, "scanned"), page(2, "scanned"), page(3, "scanned")), settings())
    assert plan.strategy == "per_page"
    assert plan.reason == "adaptive: rows unknown (no text layer), 3 item pages > 2 -> per_page"
    mixed = plan_calls(doc(page(1, text=rows_text(1, 5)), page(2, "scanned")), settings())
    assert mixed.strategy == "two_call"
    assert "2 item pages <= 2 (1 without text, rows unknown)" in mixed.reason


# --- Against the synthetic dataset -----------------------------------------------------------


def manifest() -> list[dict[str, str]]:
    with (DATASET / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@needs_dataset
def test_item_pages_and_rows_match_truth_on_native_docs() -> None:
    s = settings(native_mode="text")  # text only: no rendering needed
    for row in (r for r in manifest() if r["variant"] == "N"):
        meta = json.loads((DATASET / row["truth"]).read_text(encoding="utf-8"))["meta"]
        plan = plan_calls(prepare_sync((DATASET / row["pdf"]).read_bytes(), s), s)
        truth_pages = [n for n, lines in enumerate(meta["lines_per_page"], start=1) if lines]
        assert plan.item_pages == truth_pages, row["id"]
        assert 0 <= plan.estimated_rows - int(row["line_count"]) <= 1, row["id"]


@needs_dataset
def test_scanned_pages_are_always_item_pages() -> None:
    s = settings(image_dpi=24)
    for row in (r for r in manifest() if r["variant"] != "N"):
        d = prepare_sync((DATASET / row["pdf"]).read_bytes(), s)
        scanned = {p.page_no for p in d.kept_pages if p.kind == "scanned"}
        assert scanned and scanned <= set(item_pages(d)), row["id"]
