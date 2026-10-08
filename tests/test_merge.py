"""Tests for merging call results (Step 3.6). No LLM, no network."""

import time
from datetime import date
from decimal import Decimal

from app.extraction.llm_client import CallResult, Timings, Tokens
from app.extraction.merge import RowBatch, merge, merge_outcomes
from app.extraction.orchestrator import CallOutcome, TimelineEntry
from app.extraction.planner import CallSpec
from app.extraction.prompts import header_task, lines_task

# Compact rows in LLM_LINE_ITEM_COLUMNS order:
# line_no, item_code, description, hsn_sac, quantity, uom, unit_rate, discount_pct,
# taxable_value, gst_rate, line_total, line_delivery_date


def row(n: int, description: str = "Hex bolt", **cells) -> list:
    values = {
        "line_no": n,
        "item_code": f"BLT-{n}",
        "description": description,
        "hsn_sac": "731815",
        "quantity": 10,
        "uom": "NOS",
        "unit_rate": Decimal("100"),
        "discount_pct": None,
        "taxable_value": Decimal("1000.00"),
        "gst_rate": 18,
        "line_total": Decimal("1180.00"),
        "line_delivery_date": None,
    }
    values.update(cells)
    return list(values.values())


def test_two_call_rows_become_line_items() -> None:
    merged = merge({"po_number": "PO/1"}, [RowBatch("LI", None, [row(1), row(2)])])
    assert [item.line_no for item in merged.items] == [1, 2]
    assert merged.items[0].quantity == Decimal("10") and merged.items[0].gst_rate == Decimal(18)
    assert merged.row_sources == {1: "LI", 2: "LI"}
    assert merged.row_errors == [] and merged.line_calls == ["LI"]


def test_per_page_batches_are_concatenated_in_page_order() -> None:
    batches = [RowBatch("LI-p2", 2, [row(3)]), RowBatch("LI-p1", 1, [row(1), row(2)])]
    merged = merge(None, sorted(batches, key=lambda b: b.page_no))
    assert [item.line_no for item in merged.items] == [1, 2, 3]
    assert merged.row_sources == {1: "LI-p1", 2: "LI-p1", 3: "LI-p2"}


def test_continuation_fragment_is_appended_to_the_previous_row() -> None:
    first = row(2, "Deep groove ball bearing 6205-2RS,", taxable_value=None, line_total=None)
    fragment = [
        None,
        None,
        "25 x 52 x 15 mm, sealed",
        None,
        None,
        None,
        None,
        None,
        Decimal("4040.00"),
        None,
        Decimal("4767.20"),
        None,
    ]
    merged = merge(
        None, [RowBatch("LI-p1", 1, [row(1), first]), RowBatch("LI-p2", 2, [fragment, row(3)])]
    )
    assert [item.line_no for item in merged.items] == [1, 2, 3]
    joined = merged.items[1]
    assert joined.description == "Deep groove ball bearing 6205-2RS, 25 x 52 x 15 mm, sealed"
    assert joined.taxable_value == Decimal("4040.00") and joined.line_total == Decimal("4767.20")
    assert merged.row_sources[2] == "LI-p1" and merged.row_errors == []
    assert any("continuation" in note for note in merged.notes)


def test_duplicate_line_no_keeps_the_most_complete_row() -> None:
    partial = row(2, "Bearing", item_code=None, uom=None)
    complete = row(2, "Bearing 6205-2RS, 25 x 52 x 15 mm")
    batches = [RowBatch("LI-p1", 1, [row(1), partial]), RowBatch("LI-p2", 2, [complete, row(3)])]
    merged = merge(None, batches)
    assert [item.line_no for item in merged.items] == [1, 2, 3]
    assert merged.items[1].description == "Bearing 6205-2RS, 25 x 52 x 15 mm"
    assert merged.row_sources[2] == "LI-p2"
    assert any("line 2 given twice" in note for note in merged.notes)
    # "2" and 2 are the same line number
    merged = merge(None, [RowBatch("LI", None, [row(1), row(2), row("2", item_code=None)])])
    assert [item.line_no for item in merged.items] == [1, 2]


def test_bad_rows_become_row_errors_with_their_call() -> None:
    batches = [
        RowBatch("LI-p1", 1, [row(1), row(2, quantity="ten")]),
        RowBatch("LI-p2", 2, [[3, "BLT-3"], "not a row", row(4, description=None)]),
    ]
    merged = merge(None, batches)
    assert [item.line_no for item in merged.items] == [1]
    errors = [(e.call_id, e.line_no) for e in merged.row_errors]
    assert errors == [("LI-p2", None), ("LI-p2", None), ("LI-p1", 2), ("LI-p2", 4)]
    assert "expected 12 cells, got 2" in merged.row_errors[0].message
    assert "quantity" in merged.row_errors[2].message


def test_header_is_validated_field_by_field() -> None:
    merged = merge(
        {
            "po_number": "PO/1",
            "po_date": "14/03/2026",
            "grand_total": "abc",
            "delivery_date": "31/02/2026",
            "subtotal": Decimal("1950.00"),
            "made_up": 1,
        },
        [],
    )
    assert merged.header["po_date"] == date(2026, 3, 14)  # parsed day-first
    assert merged.header["subtotal"] == Decimal("1950.00")
    assert set(merged.header_errors) == {"grand_total", "delivery_date", "made_up"}
    assert "grand_total" not in merged.header


def test_absent_tax_totals_are_zero_but_only_with_a_header() -> None:
    merged = merge({"po_number": "PO/1", "igst_total": Decimal("90.00")}, [])
    assert merged.header["cgst_total"] == merged.header["sgst_total"] == Decimal("0.00")
    assert merged.header["igst_total"] == Decimal("90.00")
    assert len(merged.notes) == 2
    assert "cgst_total" not in merge(None, []).header  # no header result: nothing defaulted


def outcome(call_id: str, task, parsed=None, error=None) -> CallOutcome:
    now = time.time()
    result = CallResult(
        call_id,
        "po-fast",
        parsed,
        "{}",
        None,
        Tokens(),
        Timings(now, now, 0),
        1,
        error=error,
        error_kind="transient" if error else None,
    )
    spec = CallSpec(call_id, task, "po-fast", 100)
    return CallOutcome(spec, result, TimelineEntry(call_id, "po-fast", 0, 0, 1, error is None))


def test_merge_outcomes_lists_failed_calls() -> None:
    outcomes = [
        outcome("H", header_task(), {"po_number": "PO/1"}),
        outcome("LI-p2", lines_task(2), {"rows": [row(3)]}),
        outcome("LI-p1", lines_task(1), {"rows": [row(1), row(2)]}),
        outcome("LI-p3", lines_task(3), error="HTTP 503"),
        outcome("LI-p4", lines_task(4), {"not_rows": []}),
    ]
    merged = merge_outcomes(outcomes)
    assert merged.header["po_number"] == "PO/1"
    assert [item.line_no for item in merged.items] == [1, 2, 3]
    assert merged.missing_calls == ["LI-p3", "LI-p4"]
    assert merged.line_calls == ["LI-p1", "LI-p2", "LI-p3", "LI-p4"]
