"""Tests for the synthetic PO generator and its consistency self-check (Step 1.3)."""

import json
import re
from collections import Counter
from decimal import Decimal

import pytest

from datagen.consistency import check_po, state_code_from_address, state_code_from_place
from datagen.generate import DROPPABLE_FIELDS, GeneratedPO, Knobs, generate_many, generate_po
from datagen.india import pan_from_gstin
from schema.po_schema import PurchaseOrder


@pytest.fixture(scope="module")
def dataset() -> list[GeneratedPO]:
    return generate_many(200, seed=2026)


# --- Ground truth is exactly consistent ------------------------------------------------------


def test_200_generated_pos_have_zero_violations(dataset: list[GeneratedPO]) -> None:
    violations = {g.meta.seed: check_po(g.po) for g in dataset}
    assert {seed: v for seed, v in violations.items() if v} == {}


def test_tax_regime_matches_meta(dataset: list[GeneratedPO]) -> None:
    for g in dataset:
        po = g.po
        if g.meta.inter_state:
            assert po.igst_total > 0 and po.cgst_total == po.sgst_total == 0
        else:
            assert po.igst_total == 0 and po.cgst_total == po.sgst_total > 0
        assert state_code_from_place(po.place_of_supply) == state_code_from_address(
            po.ship_to_address
        )


def test_related_parties_share_the_buyer_pan(dataset: list[GeneratedPO]) -> None:
    for g in dataset:
        po = g.po
        buyer_pan = pan_from_gstin(po.buyer_gstin)
        for gstin in (po.bill_to_gstin, po.ship_to_gstin):
            if gstin is not None:
                assert pan_from_gstin(gstin) == buyer_pan


def test_grand_total_is_whole_rupees_and_round_off_bounded(dataset: list[GeneratedPO]) -> None:
    for g in dataset:
        assert g.po.grand_total == g.po.grand_total.to_integral_value()
        assert abs(g.po.round_off) <= Decimal("0.50")


def test_line_numbers_and_count_range(dataset: list[GeneratedPO]) -> None:
    for g in dataset:
        numbers = [item.line_no for item in g.po.line_items]
        assert numbers == list(range(1, len(numbers) + 1))
        assert 30 <= len(numbers) <= 60


def test_item_codes_unique_within_po(dataset: list[GeneratedPO]) -> None:
    for g in dataset:
        codes = [item.item_code for item in g.po.line_items if item.item_code]
        assert len(codes) == len(set(codes))


# --- Determinism -------------------------------------------------------------------------------


def test_same_seed_same_po() -> None:
    first = generate_po(123)
    generate_many(5, seed=9)  # other generation in between must not leak state
    assert generate_po(123) == first
    assert generate_po(124) != first


def test_same_seed_same_dataset() -> None:
    assert generate_many(10, seed=1) == generate_many(10, seed=1)
    assert generate_many(10, seed=1) != generate_many(10, seed=2)


def test_each_po_can_be_rebuilt_from_its_seed() -> None:
    for g in generate_many(5, seed=3):
        assert generate_po(g.meta.seed) == g


# --- Knobs -------------------------------------------------------------------------------------


def test_inter_state_mix_matches_knob(dataset: list[GeneratedPO]) -> None:
    share = sum(g.meta.inter_state for g in dataset) / len(dataset)
    assert abs(share - Knobs().p_inter_state) < 0.08


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_inter_state_extremes(p: float) -> None:
    pos = generate_many(20, seed=4, knobs=Knobs(p_inter_state=p))
    assert {g.meta.inter_state for g in pos} == {bool(p)}
    assert all(check_po(g.po) == [] for g in pos)


def test_optional_drop_rate_matches_knob(dataset: list[GeneratedPO]) -> None:
    absent = sum(getattr(g.po, name) is None for g in dataset for name in DROPPABLE_FIELDS)
    share = absent / (len(dataset) * len(DROPPABLE_FIELDS))
    assert abs(share - Knobs().optional_drop_rate) < 0.03


def test_no_optional_fields_dropped_when_rate_is_zero() -> None:
    for g in generate_many(10, seed=5, knobs=Knobs(optional_drop_rate=0)):
        assert all(getattr(g.po, name) is not None for name in DROPPABLE_FIELDS)
        assert all(item.item_code for item in g.po.line_items)


def test_all_optional_knobs_on() -> None:
    knobs = Knobs(
        p_amendment=1, p_quotation_ref=1, p_indent_no=1, p_freight=1, p_other_charges=1,
        p_header_discount=1, p_line_delivery_dates=1, p_discount_per_line=1,
        p_bill_to_differs=1, p_ship_to_differs=1, p_tc_page=1, optional_drop_rate=0,
    )  # fmt: skip
    for g in generate_many(20, seed=6, knobs=knobs):
        po = g.po
        assert check_po(po) == []
        assert po.amendment_no and po.amendment_date and po.quotation_ref and po.indent_no
        assert po.freight_charges and po.other_charges and po.discount_total
        assert all(i.line_delivery_date and i.discount_pct for i in po.line_items)
        assert po.bill_to_address != po.buyer_address
        assert po.ship_to_address != po.bill_to_address
        assert g.meta.has_tc_page


def test_all_optional_knobs_off() -> None:
    knobs = Knobs(
        p_amendment=0, p_quotation_ref=0, p_indent_no=0, p_freight=0, p_other_charges=0,
        p_header_discount=0, p_line_delivery_dates=0, p_discount_per_line=0,
        p_multiline_description_per_line=0, p_bill_to_differs=0, p_ship_to_differs=0,
        p_tc_page=0,
    )  # fmt: skip
    for g in generate_many(20, seed=7, knobs=knobs):
        po = g.po
        assert check_po(po) == []
        assert po.amendment_no is po.quotation_ref is po.indent_no is None
        assert po.freight_charges is po.other_charges is po.discount_total is None
        assert all(i.discount_pct is None and i.line_delivery_date is None for i in po.line_items)
        assert po.bill_to_address == po.ship_to_address == po.buyer_address
        assert not g.meta.has_tc_page


def test_line_count_knob() -> None:
    for g in generate_many(10, seed=8, knobs=Knobs(min_lines=3, max_lines=5)):
        assert 3 <= len(g.po.line_items) <= 5


@pytest.mark.parametrize(
    "kwargs",
    [{"min_lines": 0}, {"min_lines": 10, "max_lines": 5}, {"p_inter_state": 1.5},
     {"optional_drop_rate": -0.1}],
)  # fmt: skip
def test_invalid_knobs_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Knobs(**kwargs)


def test_po_number_formats_vary(dataset: list[GeneratedPO]) -> None:
    patterns = {
        "fy_slash": r"PO/\d{4}-\d{2}/\d{5}",
        "sap": r"45000\d{5}",
        "dept": r"[A-Z]{3}-PUR-\d{2}-\d{4}",
        "unit_fy": r"[A-Z]{3}/PO/\d{1,4}/\d{2}-\d{2}",
        "yearmonth": r"PO-\d{6}-\d{4}",
    }
    seen = Counter(
        next(name for name, rx in patterns.items() if re.fullmatch(rx, g.po.po_number))
        for g in dataset
    )
    assert set(seen) == set(patterns)


# --- JSON round-trip ---------------------------------------------------------------------------


def test_json_round_trip_preserves_decimals(dataset: list[GeneratedPO]) -> None:
    for g in dataset[:20]:
        text = g.po.model_dump_json()
        back = PurchaseOrder.model_validate_json(text)
        assert back == g.po
        for original, restored in zip(g.po.line_items, back.line_items, strict=True):
            # Same value AND same scale: 455.00 must not come back as 455 or 455.0.
            assert str(restored.unit_rate) == str(original.unit_rate)
            assert str(restored.taxable_value) == str(original.taxable_value)
        assert str(back.grand_total) == str(g.po.grand_total)


def test_json_never_contains_floats(dataset: list[GeneratedPO]) -> None:
    data = json.loads(dataset[0].po.model_dump_json(), parse_float=lambda s: pytest.fail(s))
    assert isinstance(data["grand_total"], str)
    assert isinstance(data["line_items"][0]["unit_rate"], str)


# --- The self-check catches errors (it is not vacuous) -----------------------------------------


@pytest.fixture
def intra_po() -> PurchaseOrder:
    knobs = Knobs(p_inter_state=0, optional_drop_rate=0, p_ship_to_differs=0)
    return generate_po(11, knobs).po


def _with(po: PurchaseOrder, line: int | None = None, **changes: object) -> PurchaseOrder:
    """Copy of `po` with header fields (or fields of line `line`) changed, no re-validation."""
    if line is None:
        return po.model_copy(update=changes)
    items = list(po.line_items)
    items[line - 1] = items[line - 1].model_copy(update=changes)
    return po.model_copy(update={"line_items": items})


def test_check_po_detects_errors(intra_po: PurchaseOrder) -> None:
    po = intra_po
    first = po.line_items[0]
    cases = {
        "taxable_value": _with(po, 1, taxable_value=first.taxable_value + Decimal("0.01")),
        "cgst/sgst": _with(po, 1, cgst_amount=first.cgst_amount + Decimal("0.01")),
        "must not carry IGST": _with(po, 1, igst_amount=Decimal("1.00")),
        "line_total": _with(po, 1, line_total=first.line_total + 1),
        "subtotal": _with(po, subtotal=po.subtotal + 1),
        "grand_total": _with(po, grand_total=po.grand_total + 1),
        "round_off": _with(po, round_off=Decimal("0.75")),
        "amount_in_words": _with(po, amount_in_words="Rupees One Only"),
        "invalid GSTIN": _with(po, vendor_gstin=po.vendor_gstin[:14] + "0"),
        "PAN": _with(po, buyer_pan="AAAAA0000A"),
        "state_code": _with(po, buyer_state_code="01"),
        "place_of_supply": _with(po, place_of_supply="Ladakh (38)"),
        "before po_date": _with(po, delivery_date=po.po_date.replace(year=po.po_date.year - 1)),
        "not sequential": _with(po, 2, line_no=7),
    }
    for expected, broken in cases.items():
        violations = check_po(broken)
        assert any(expected in v for v in violations), (expected, violations)


def test_state_code_helpers() -> None:
    assert state_code_from_place("Maharashtra (27)") == "27"
    assert state_code_from_place("27-Maharashtra") == "27"
    assert state_code_from_place("Tamil Nadu") == "33"
    assert state_code_from_place("Nowhere") is None
    # A plot number must not be mistaken for a state code.
    assert state_code_from_address("Plot No. 33, MIDC Bhosari, Pune - 411026, Maharashtra") == "27"
