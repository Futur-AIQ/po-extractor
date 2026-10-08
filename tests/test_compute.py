"""Tests for computed fields (Step 3.6). No LLM, no network.

The dataset test reads data/synthetic (built by `make dataset`) and is skipped without it.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from app.common.gst import gstin_check_char
from app.extraction.compute import compute, supply_type
from app.extraction.merge import RowBatch, merge
from schema.fields import COMPUTED_FIELDS, LLM_LINE_ITEM_COLUMNS

DATASET = Path("data/synthetic")
needs_dataset = pytest.mark.skipif(
    not (DATASET / "manifest.csv").exists(), reason="data/synthetic not built (make dataset)"
)


def gstin(state: str) -> str:
    """A checksum-valid GSTIN in `state`."""
    first14 = f"{state}AAPFU0939F1Z"
    return first14 + gstin_check_char(first14)


def rows(*taxable_and_rate: tuple[str, int]) -> list[RowBatch]:
    cells = []
    for n, (taxable, rate) in enumerate(taxable_and_rate, start=1):
        cells.append(
            [
                n,
                None,
                "Item",
                "731815",
                1,
                "NOS",
                Decimal(taxable),
                None,
                Decimal(taxable),
                rate,
                Decimal(taxable),
                None,
            ]
        )
    return [RowBatch("LI", None, cells)]


def test_supply_type_from_vendor_gstin_and_place_of_supply() -> None:
    assert supply_type({"vendor_gstin": gstin("29"), "place_of_supply": "29-Karnataka"}) == "intra"
    assert supply_type({"vendor_gstin": gstin("27"), "place_of_supply": "Karnataka"}) == "inter"
    assert (
        supply_type({"vendor_gstin": gstin("27"), "place_of_supply": "Maharashtra (27)"}) == "intra"
    )


def test_supply_type_fallbacks() -> None:
    # Vendor state from the printed state code when the GSTIN is invalid.
    assert (
        supply_type(
            {
                "vendor_gstin": "29BADGSTIN",
                "vendor_state_code": "29",
                "place_of_supply": "Karnataka",
            }
        )
        == "intra"
    )
    # Place of supply from the ship-to state code, then the ship-to GSTIN.
    assert supply_type({"vendor_gstin": gstin("29"), "ship_to_state_code": "27"}) == "inter"
    assert supply_type({"vendor_gstin": gstin("29"), "ship_to_gstin": gstin("29")}) == "intra"
    # Unknown -> None, never a guess.
    assert supply_type({"vendor_gstin": gstin("29")}) is None
    assert supply_type({"place_of_supply": "Karnataka"}) is None


def test_intra_state_splits_tax_half_up() -> None:
    header = {
        "vendor_gstin": gstin("29"),
        "place_of_supply": "29-Karnataka",
        "cgst_total": Decimal("90.09"),
        "sgst_total": Decimal("90.09"),
    }
    merged = compute(merge(header, rows(("1000.00", 18), ("2.50", 5))))
    assert merged.supply_type == "intra"
    first, second = merged.items
    assert (first.cgst_amount, first.sgst_amount, first.igst_amount) == (Decimal("90.00"),) * 2 + (
        None,
    )
    assert second.cgst_amount == second.sgst_amount == Decimal("0.06")
    assert merged.header["total_tax"] == Decimal("180.18")  # printed totals, igst taken as 0


def test_half_up_rounding_on_the_half_paisa() -> None:
    merged = compute(
        merge({"vendor_gstin": gstin("29"), "place_of_supply": "Karnataka"}, rows(("2.50", 10)))
    )
    assert merged.items[0].cgst_amount == Decimal("0.13")  # 0.125 rounds half up, not to even


def test_inter_state_is_igst_only() -> None:
    merged = compute(
        merge({"vendor_gstin": gstin("27"), "place_of_supply": "Karnataka"}, rows(("1234.56", 18)))
    )
    item = merged.items[0]
    assert merged.supply_type == "inter"
    assert (item.cgst_amount, item.sgst_amount, item.igst_amount) == (None, None, Decimal("222.22"))


def test_unknown_supply_type_computes_no_line_taxes() -> None:
    merged = compute(merge({"vendor_gstin": gstin("29")}, rows(("100.00", 18))))
    assert merged.supply_type is None
    assert merged.items[0].cgst_amount is None and merged.items[0].igst_amount is None


@needs_dataset
def test_computed_fields_reproduce_the_truth_exactly() -> None:
    for path in sorted((DATASET / "truth").glob("*.json")):
        po = json.loads(path.read_text(encoding="utf-8"))["po"]
        header = {
            k: v
            for k, v in po.items()
            if k != "line_items" and k not in COMPUTED_FIELDS and v is not None
        }
        cells = [[line.get(c) for c in LLM_LINE_ITEM_COLUMNS] for line in po["line_items"]]
        merged = compute(merge(header, [RowBatch("LI", None, cells)]))
        for item, truth in zip(merged.items, po["line_items"], strict=True):
            for name in ("cgst_amount", "sgst_amount", "igst_amount"):
                value = getattr(item, name)
                assert (str(value) if value is not None else None) == truth[name], (path.name, name)
        if po["total_tax"] is not None:  # the truth leaves it out where it is not printed
            assert str(merged.header["total_tax"]) == po["total_tax"], path.name
