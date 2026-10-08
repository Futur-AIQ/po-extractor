"""Tests for the validation engine (Step 3.6): one pass and one fail case per rule.

The dataset test reads data/synthetic (built by `make dataset`) and is skipped without it.
"""

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.common.gst import gstin_check_char
from app.common.money import ZERO, round_money
from app.core.settings import Settings
from app.extraction.compute import compute
from app.extraction.grounding import GroundingIndex
from app.extraction.merge import MergedPO, RowBatch, merge
from app.extraction.preprocess import prepare_sync
from app.extraction.validate import (
    Masters,
    ValidationReport,
    load_masters,
    to_purchase_order,
    validate,
)
from datagen.india import amount_in_words_inr
from schema.fields import COMPUTED_FIELDS, LLM_LINE_ITEM_COLUMNS

DATASET = Path("data/synthetic")
needs_dataset = pytest.mark.skipif(
    not (DATASET / "manifest.csv").exists(), reason="data/synthetic not built (make dataset)"
)


def gstin(state: str, pan: str) -> str:
    first14 = f"{state}{pan}1Z"
    return first14 + gstin_check_char(first14)


BUYER_PAN, VENDOR_PAN = "AAPFU0939F", "BBBCV1234K"
BUYER, VENDOR = gstin("29", BUYER_PAN), gstin("29", VENDOR_PAN)

# (qty, rate, discount %, GST %) for lines 1-5; lines 1-2 on page 1, lines 3-5 on page 2.
LINES = [
    (10, "100", None, 18),
    (5, "200", 10, 12),
    (1, "50", None, 5),
    (4, "25.50", None, 18),
    (2, "1000", 5, 28),
]
PAGES = {"LI-p1": [1, 2], "LI-p2": [3, 4, 5]}


def row(n: int) -> list[Any]:
    qty, rate, discount, gst = LINES[n - 1]
    taxable = round_money(qty * Decimal(rate) * (1 - Decimal(discount or 0) / 100))
    half = round_money(taxable * gst / 2 / 100)
    return [
        n,
        f"BLT/{n:04d}",
        f"Item {n}",
        "731815",
        qty,
        "NOS",
        Decimal(rate),
        discount,
        taxable,
        gst,
        taxable + 2 * half,
        None,
    ]


def header(rows: list[list[Any]]) -> dict[str, Any]:
    """A consistent intra-state header for `rows` (taxes as code computes them)."""
    subtotal = sum((r[8] for r in rows), ZERO)
    cgst = sum((round_money(r[8] * r[9] / 2 / 100) for r in rows), ZERO)
    before = subtotal + 2 * cgst + Decimal("100.00")  # freight 100
    grand = before.quantize(Decimal("1")).quantize(Decimal("0.01"))
    return {
        "po_number": "PO/2025-26/00042",
        "po_date": "03/04/2026",
        "delivery_date": "30/04/2026",
        "currency": "INR",
        "buyer_name": "Acme Industries",
        "buyer_gstin": BUYER,
        "buyer_pan": BUYER_PAN,
        "buyer_state": "Karnataka",
        "buyer_state_code": "29",
        "vendor_name": "Bolt Works",
        "vendor_gstin": VENDOR,
        "vendor_pan": VENDOR_PAN,
        "place_of_supply": "29-Karnataka",
        "subtotal": subtotal,
        "freight_charges": Decimal("100.00"),
        "cgst_total": cgst,
        "sgst_total": cgst,
        "igst_total": ZERO,
        "round_off": grand - before,
        "grand_total": grand,
        "amount_in_words": amount_in_words_inr(grand),
    }


def build(
    edit_header: dict[str, Any] | None = None,
    edit_rows: dict[int, dict[int, Any]] | None = None,
    drop: tuple[int, ...] = (),
    drop_header: tuple[str, ...] = (),
) -> MergedPO:
    """Merged + computed PO; edits are applied to the simulated LLM output."""
    rows = {n: row(n) for n in range(1, len(LINES) + 1)}
    h = header(list(rows.values()))
    h.update(edit_header or {})
    for name in drop_header:
        h.pop(name)
    for n, cells in (edit_rows or {}).items():
        for column, value in cells.items():
            rows[n][column] = value
    batches = [
        RowBatch(call, int(call[-1]), [rows[n] for n in numbers if n not in drop])
        for call, numbers in PAGES.items()
    ]
    return compute(merge(h, batches))


def text_of(merged: MergedPO) -> str:
    h = merged.header
    values = [h[k] for k in ("po_number", "buyer_gstin", "vendor_gstin", "buyer_pan", "vendor_pan")]
    values += [v for item in merged.items for v in (item.item_code, item.hsn_sac)]
    return "\n".join(str(v) for v in values)


MASTERS = Masters(
    party_gstins=frozenset({BUYER, VENDOR}),
    item_codes_by_pan={BUYER_PAN: frozenset(f"BLT/{n:04d}" for n in range(1, 6))},
    processed_by_pan={BUYER_PAN: frozenset({"PO2025260001"})},
)


def run(merged: MergedPO, index: GroundingIndex | None = None, masters=MASTERS) -> ValidationReport:
    index = index or GroundingIndex([text_of(build())], complete=True)
    return validate(merged, index, masters)


def check(report: ValidationReport, rule: str):
    return next(c for c in report.checks if c.rule == rule)


def failed_only(report: ValidationReport, rule: str):
    """Assert that `rule` is the only failed check and return it."""
    assert [c.rule for c in report.failed] == [rule], [(c.rule, c.detail) for c in report.failed]
    return check(report, rule)


# --- All rules pass ----------------------------------------------------------------------


def test_consistent_po_passes_every_rule() -> None:
    merged = build()
    report = run(merged)
    assert report.passed, [(c.rule, c.detail) for c in report.failed]
    assert [c.rule for c in report.checks] == [
        "schema",
        "gstin",
        "tax_regime",
        "line_arithmetic",
        "totals",
        "continuity",
        "amount_in_words",
        "formats",
        "grounding",
        "masters",
        "duplicate_po",
    ]
    po = to_purchase_order(merged)
    assert po is not None and len(po.line_items) == 5 and po.total_tax == 2 * po.cgst_total


# --- Rule 1: schema ----------------------------------------------------------------------


def test_schema_missing_critical_header_field_retries_header() -> None:
    merged = build(drop_header=("grand_total",))
    c = check(run(merged), "schema")
    assert not c.passed and c.fields == ("grand_total",) and c.retry_target == "H"
    assert c.retryable and to_purchase_order(merged) is None


def test_schema_invalid_row_retries_its_call_and_both_when_header_too() -> None:
    c = check(run(build(edit_rows={4: {4: "four"}})), "schema")
    assert not c.passed and c.fields == ("line_items[4]",) and c.retry_target == "LI-p2"
    both = build(edit_rows={4: {4: "four"}}, edit_header={"po_date": "32/13/2026"})
    c = check(run(both), "schema")
    assert c.retry_target == "both" and "po_date" in c.fields


def test_schema_failed_call_is_reported() -> None:
    merged = build()
    merged.missing_calls = ["LI-p2"]
    c = check(run(merged), "schema")
    assert not c.passed and c.retry_target == "LI-p2" and "LI-p2: no result" in c.detail


# --- Rule 2: GSTIN and PAN ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "fields"),
    [
        ({"vendor_gstin": VENDOR[:14] + ("A" if VENDOR[14] != "A" else "B")}, ("vendor_gstin",)),
        ({"buyer_state_code": "27"}, ("buyer_state_code", "buyer_gstin")),
        ({"buyer_state": "Maharashtra"}, ("buyer_state", "buyer_gstin")),
        ({"vendor_pan": "BBBCV9999K"}, ("vendor_pan", "vendor_gstin")),
        ({"buyer_pan": "AAAAA1234A"}, ("buyer_pan", "buyer_gstin")),  # not the GSTIN's PAN
        ({"buyer_pan": "AAADA1234A"}, ("buyer_pan",)),  # 'D' is not a PAN holder type
    ],
)
def test_gstin_rule_failures(edit: dict, fields: tuple[str, ...]) -> None:
    report = run(build(edit_header=edit))
    c = check(report, "gstin")
    assert not c.passed and c.retry_target == "H" and c.retryable
    assert set(fields) <= set(c.fields)


# --- Rule 3: tax regime ------------------------------------------------------------------


def test_tax_regime_intra_state_with_igst_fails() -> None:
    c = check(run(build(edit_header={"igst_total": Decimal("10.00")})), "tax_regime")
    assert not c.passed and "igst_total" in c.fields and c.retry_target == "H"


def test_tax_regime_unknown_supply_type_fails() -> None:
    c = check(run(build(drop_header=("place_of_supply",))), "tax_regime")
    assert not c.passed and c.fields == ("vendor_gstin", "place_of_supply")


def test_tax_regime_inter_state_passes_with_igst_only() -> None:
    merged = build(edit_header={"place_of_supply": "Maharashtra (27)"})
    assert merged.supply_type == "inter"
    assert check(run(merged), "tax_regime").passed is False  # printed CGST/SGST on inter-state
    h = header([row(n) for n in range(1, 6)])
    igst = h["cgst_total"] * 2
    merged = build(
        edit_header={
            "place_of_supply": "Maharashtra (27)",
            "cgst_total": ZERO,
            "sgst_total": ZERO,
            "igst_total": igst,
        }
    )
    assert check(run(merged), "tax_regime").passed


# --- Rule 4: line arithmetic -------------------------------------------------------------


def test_line_arithmetic_wrong_taxable_value_retries_that_page() -> None:
    c = check(run(build(edit_rows={3: {8: Decimal("60.00")}})), "line_arithmetic")
    assert not c.passed and c.retry_target == "LI-p2"
    assert "line_items[3].taxable_value" in c.fields


def test_line_arithmetic_wrong_gst_rate_breaks_line_total() -> None:
    c = check(run(build(edit_rows={1: {9: 12}})), "line_arithmetic")
    assert not c.passed and c.retry_target == "LI-p1"
    assert c.fields == ("line_items[1].line_total", "line_items[1].gst_rate")


def test_line_arithmetic_tolerates_one_rupee() -> None:
    assert check(run(build(edit_rows={2: {8: Decimal("900.50")}})), "line_arithmetic").passed


# --- Rule 5: totals ----------------------------------------------------------------------


def test_totals_lines_vs_subtotal_retries_both() -> None:
    c = check(run(build(edit_header={"subtotal": Decimal("99.00")})), "totals")
    assert not c.passed and c.retry_target == "both" and "subtotal" in c.fields


def test_totals_header_only_formula_retries_header() -> None:
    h = header([row(n) for n in range(1, 6)])
    c = check(run(build(edit_header={"grand_total": h["grand_total"] + 50})), "totals")
    assert not c.passed and c.retry_target == "H" and c.fields == ("grand_total",)


def test_totals_round_off_limit() -> None:
    c = check(run(build(edit_header={"round_off": Decimal("1.50")})), "totals")
    assert not c.passed and "round_off" in c.fields


def test_totals_dropped_last_row_is_caught() -> None:
    report = run(build(drop=(5,)))
    assert check(report, "continuity").passed  # no gap: rows 1-4 are continuous
    c = check(report, "totals")
    assert not c.passed and c.retry_target == "both"


# --- Rule 6: continuity ------------------------------------------------------------------


def test_continuity_gap_inside_one_page_retries_that_page() -> None:
    c = check(run(build(drop=(4,))), "continuity")
    assert not c.passed and c.fields == ("line_items[4]",) and c.retry_target == "LI-p2"


def test_continuity_gap_at_a_page_break_retries_both_neighbours() -> None:
    c = check(run(build(drop=(3,))), "continuity")
    assert not c.passed and c.retry_target == "LI"  # rows 2 (p1) and 4 (p2) around the gap


def test_continuity_accepts_steps_of_ten() -> None:
    rows = [row(n) for n in (1, 2, 3)]
    for index, r in enumerate(rows, start=1):
        r[0] = index * 10
    merged = compute(merge(header(rows), [RowBatch("LI", None, rows)]))
    assert check(run(merged), "continuity").passed


def test_continuity_no_rows() -> None:
    c = check(run(compute(merge(header([]), []))), "continuity")
    assert not c.passed and c.retry_target == "LI"


# --- Rule 7: amount in words -------------------------------------------------------------


def test_amount_in_words_mismatch_and_unreadable() -> None:
    c = check(run(build(edit_header={"amount_in_words": "Rupees Ten Only"})), "amount_in_words")
    assert not c.passed and c.fields == ("amount_in_words", "grand_total") and c.retry_target == "H"
    c = check(run(build(edit_header={"amount_in_words": "Rupees Many Only"})), "amount_in_words")
    assert not c.passed and c.fields == ("amount_in_words",)


def test_amount_in_words_absent_or_without_paise_passes() -> None:
    assert check(run(build(drop_header=("amount_in_words",))), "amount_in_words").passed
    total = header([row(n) for n in range(1, 6)])["grand_total"]
    words = amount_in_words_inr(total + Decimal("0.40")).split(" and ")[0] + " Only"
    merged = build(edit_header={"grand_total": total + Decimal("0.40"), "amount_in_words": words})
    assert check(run(merged), "amount_in_words").passed


# --- Rule 8: formats ---------------------------------------------------------------------


def test_formats_dates_currency_and_hsn() -> None:
    c = check(run(build(edit_header={"delivery_date": "01/03/2026"})), "formats")
    assert not c.passed and c.fields == ("delivery_date",) and c.retry_target == "H"
    c = check(run(build(edit_header={"currency": "RUPEES"})), "formats")
    assert not c.passed and c.fields == ("currency",)
    c = check(run(build(edit_rows={5: {3: "73A8"}})), "formats")
    assert not c.passed and c.fields == ("line_items[5].hsn_sac",) and c.retry_target == "LI-p2"
    merged = build(edit_header={"quotation_date": "05/04/2026"}, edit_rows={1: {3: "12"}})
    c = check(run(merged), "formats")
    assert c.retry_target == "both"


# --- Rule 9: grounding -------------------------------------------------------------------


def test_grounding_value_not_in_text_layer_fails() -> None:
    merged = build()
    text = text_of(merged).replace(VENDOR, "").replace("BLT/0004", "")
    report = run(merged, GroundingIndex([text], complete=True))
    c = check(report, "grounding")
    assert not c.passed and c.retry_target == "both"
    assert c.fields == ("vendor_gstin", "line_items[4].item_code")


def test_grounding_unverifiable_passes_with_a_note() -> None:
    merged = build()
    c = check(run(merged, GroundingIndex(["scanned PO, no text"], complete=False)), "grounding")
    assert c.passed and "unverifiable" in c.detail


# --- Rule 10: masters and duplicates -----------------------------------------------------


def test_masters_unknown_party_and_item_code() -> None:
    masters = Masters(frozenset({BUYER}), MASTERS.item_codes_by_pan, MASTERS.processed_by_pan)
    c = check(validate(build(), None, masters), "masters")
    assert not c.passed and c.fields == ("vendor_gstin",) and c.retry_target == "H"
    merged = build(edit_rows={2: {1: "BLT/9999"}})
    c = check(run(merged, GroundingIndex([text_of(merged)], complete=True)), "masters")
    assert not c.passed and c.fields == ("line_items[2].item_code",) and c.retry_target == "LI-p1"


def test_masters_without_item_master_passes_with_note() -> None:
    masters = Masters(MASTERS.party_gstins, {}, {})
    c = check(validate(build(), None, masters), "masters")
    assert c.passed and "no item master" in c.detail


def test_duplicate_po_is_not_retryable() -> None:
    merged = build(edit_header={"po_number": "PO/2025-26/0001"})
    index = GroundingIndex([text_of(merged)], complete=True)
    c = failed_only(run(merged, index), "duplicate_po")
    assert c.retryable is False and c.retry_target is None and c.fields == ("po_number",)


# --- Against the synthetic dataset -----------------------------------------------------------


@needs_dataset
def test_every_truth_po_passes_every_rule_except_expected_duplicates() -> None:
    settings = Settings(_env_file=None, native_mode="text", image_dpi=24)
    masters = load_masters(DATASET / "masters")
    with (DATASET / "manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 130
    for row_ in rows:
        truth = json.loads((DATASET / row_["truth"]).read_text(encoding="utf-8"))
        po, meta = truth["po"], truth["meta"]
        # Simulated LLM output: no computed fields, rows as compact arrays per page call.
        h = {
            k: v
            for k, v in po.items()
            if k != "line_items" and k not in COMPUTED_FIELDS and v is not None
        }
        cells = {
            li["line_no"]: [li.get(c) for c in LLM_LINE_ITEM_COLUMNS] for li in po["line_items"]
        }
        batches = [
            RowBatch(f"LI-p{page}", page, [cells[n] for n in numbers])
            for page, numbers in enumerate(meta["lines_per_page"], start=1)
            if numbers
        ]
        merged = compute(merge(h, batches))
        doc = prepare_sync((DATASET / row_["pdf"]).read_bytes(), settings)
        report = validate(merged, GroundingIndex.from_doc(doc), masters)
        failed = [(c.rule, c.retryable) for c in report.failed]
        expected = [("duplicate_po", False)] if row_["expected_duplicate"] == "True" else []
        assert failed == expected, (row_["id"], [(c.rule, c.detail) for c in report.failed])
        assert to_purchase_order(merged) is not None, row_["id"]
