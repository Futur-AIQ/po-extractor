"""The generator's own self-check: is a ground-truth PurchaseOrder exactly consistent?

check_po(po) recomputes every amount independently from quantities, rates and GST rates and
returns a list of human-readable violations (empty list = consistent). It implements the
arithmetic, tax and identity rules of PRD §9.4 that ground truth must satisfy exactly.

This is NOT the production validation engine (Step 3.6), which must tolerate OCR/LLM noise.
Here every comparison is exact, because generated data has no excuse to be off by a paisa.

Totals follow the PRD §9.4 formula: freight, other charges and the header discount are
adjustments in the totals block, outside the GST computation.
"""

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from datagen.india import STATES, amount_in_words_inr, is_valid_gstin
from schema.po_schema import LineItem, PurchaseOrder

PAISA = Decimal("0.01")
RUPEE = Decimal("1")
ZERO = Decimal("0.00")
MAX_ROUND_OFF = Decimal("0.50")


def round_money(value: Decimal) -> Decimal:
    """Round to 2 decimal places, half up (the convention on Indian tax documents)."""
    return value.quantize(PAISA, rounding=ROUND_HALF_UP)


def expected_taxable_value(item: LineItem) -> Decimal:
    """qty x rate x (1 - discount%/100), rounded to paise."""
    discount = item.discount_pct or Decimal(0)
    return round_money(item.quantity * item.unit_rate * (1 - discount / 100))


_STATES_BY_NAME = {state.name.lower(): state.code for state in STATES.values()}


def state_code_from_address(address: str | None) -> str | None:
    """State code of an address, from the state name it ends with (generated addresses do).

    Numbers are ignored on purpose: "Plot No. 33" must not be read as Tamil Nadu.
    """
    if not address:
        return None
    return _STATES_BY_NAME.get(address.rsplit(",", 1)[-1].strip().lower())


def state_code_from_place(place: str | None) -> str | None:
    """State code of a place of supply: '27-Maharashtra', 'Maharashtra (27)' or 'Maharashtra'."""
    if not place:
        return None
    for code in re.findall(r"\b(\d{2})\b", place):
        if code in STATES:
            return code
    name = re.sub(r"[\d()\-]", " ", place).strip().lower()
    return _STATES_BY_NAME.get(" ".join(name.split()))


def check_po(po: PurchaseOrder) -> list[str]:
    """Return every consistency violation in `po` (empty list if it is exactly consistent)."""
    inter_state = _is_inter_state(po)
    violations = _check_parties(po) + _check_dates(po)
    violations += _check_line_numbers(po)
    for item in po.line_items:
        violations += _check_line(item, inter_state)
    violations += _check_totals(po, inter_state)
    return violations


def _is_inter_state(po: PurchaseOrder) -> bool:
    """Inter-state supply when the vendor's state differs from the place of supply."""
    return po.vendor_gstin[:2] != state_code_from_place(po.place_of_supply)


# --- parties -----------------------------------------------------------------------------


def _check_party(
    label: str,
    gstin: str | None,
    address: str | None,
    state_code: str | None,
    state: str | None = None,
    pan: str | None = None,
) -> list[str]:
    """GSTIN valid; its state code agrees with the address and fields; PAN embedded in it."""
    if gstin is None:
        return []
    if not is_valid_gstin(gstin):
        return [f"{label}: invalid GSTIN {gstin}"]
    out = []
    code = gstin[:2]
    if address is not None and state_code_from_address(address) != code:
        out.append(f"{label}: GSTIN state {code} does not match address {address!r}")
    if state_code is not None and state_code != code:
        out.append(f"{label}: state_code {state_code} != GSTIN state {code}")
    if state is not None and state != STATES[code].name:
        out.append(f"{label}: state {state!r} != GSTIN state {STATES[code].name!r}")
    if pan is not None and pan != gstin[2:12]:
        out.append(f"{label}: PAN {pan} != PAN in GSTIN {gstin[2:12]}")
    return out


def _check_parties(po: PurchaseOrder) -> list[str]:
    out = _check_party(
        "buyer",
        po.buyer_gstin,
        po.buyer_address,
        po.buyer_state_code,
        po.buyer_state,
        po.buyer_pan,
    )
    out += _check_party(
        "vendor",
        po.vendor_gstin,
        po.vendor_address,
        po.vendor_state_code,
        po.vendor_state,
        po.vendor_pan,
    )
    out += _check_party("bill_to", po.bill_to_gstin, po.bill_to_address, po.bill_to_state_code)
    out += _check_party("ship_to", po.ship_to_gstin, po.ship_to_address, po.ship_to_state_code)

    supply = state_code_from_place(po.place_of_supply)
    if po.place_of_supply is not None and supply is None:
        out.append(f"place_of_supply {po.place_of_supply!r} names no known state")
    ship_to = state_code_from_address(po.ship_to_address)
    if supply is not None and ship_to is not None and supply != ship_to:
        out.append(f"place_of_supply state {supply} != ship-to state {ship_to}")
    return out


# --- dates and line numbers --------------------------------------------------------------


def _check_dates(po: PurchaseOrder) -> list[str]:
    out = []

    def not_before_po(name: str, value: date | None) -> None:
        if value is not None and value < po.po_date:
            out.append(f"{name} {value} is before po_date {po.po_date}")

    not_before_po("delivery_date", po.delivery_date)
    not_before_po("amendment_date", po.amendment_date)
    not_before_po("po_validity_date", po.po_validity_date)
    for item in po.line_items:
        not_before_po(f"line {item.line_no} line_delivery_date", item.line_delivery_date)
    if po.quotation_date is not None and po.quotation_date > po.po_date:
        out.append(f"quotation_date {po.quotation_date} is after po_date {po.po_date}")
    return out


def _check_line_numbers(po: PurchaseOrder) -> list[str]:
    numbers = [item.line_no for item in po.line_items]
    if not numbers:
        return ["PO has no line items"]
    if numbers != list(range(1, len(numbers) + 1)):
        return [f"line_no not sequential from 1: {numbers}"]
    return []


# --- line arithmetic and tax ---------------------------------------------------------------


def _check_line(item: LineItem, inter_state: bool) -> list[str]:
    tag = f"line {item.line_no}"
    out = []
    if item.quantity <= 0 or item.unit_rate <= 0:
        out.append(f"{tag}: quantity and unit_rate must be positive")
    taxable = expected_taxable_value(item)
    if item.taxable_value != taxable:
        out.append(f"{tag}: taxable_value {item.taxable_value} != expected {taxable}")

    if inter_state:
        expected_igst = round_money(item.taxable_value * item.gst_rate / 100)
        if item.igst_amount != expected_igst:
            out.append(f"{tag}: igst_amount {item.igst_amount} != expected {expected_igst}")
        if item.cgst_amount is not None or item.sgst_amount is not None:
            out.append(f"{tag}: inter-state line must not carry CGST/SGST")
        taxes = item.igst_amount or ZERO
    else:
        half = round_money(item.taxable_value * item.gst_rate / 2 / 100)
        if item.cgst_amount != half or item.sgst_amount != half:
            out.append(
                f"{tag}: cgst/sgst {item.cgst_amount}/{item.sgst_amount} != expected {half} each"
            )
        if item.igst_amount is not None:
            out.append(f"{tag}: intra-state line must not carry IGST")
        taxes = (item.cgst_amount or ZERO) + (item.sgst_amount or ZERO)

    if item.line_total != item.taxable_value + taxes:
        out.append(f"{tag}: line_total {item.line_total} != taxable + taxes")
    return out


# --- totals ----------------------------------------------------------------------------------


def _check_totals(po: PurchaseOrder, inter_state: bool) -> list[str]:
    out = []
    items = po.line_items

    def must_equal(name: str, actual: Decimal | None, expected: Decimal) -> None:
        if actual is not None and actual != expected:
            out.append(f"{name} {actual} != expected {expected}")

    must_equal("subtotal", po.subtotal, sum((i.taxable_value for i in items), ZERO))
    must_equal("cgst_total", po.cgst_total, sum((i.cgst_amount or ZERO for i in items), ZERO))
    must_equal("sgst_total", po.sgst_total, sum((i.sgst_amount or ZERO for i in items), ZERO))
    must_equal("igst_total", po.igst_total, sum((i.igst_amount or ZERO for i in items), ZERO))
    taxes = po.cgst_total + po.sgst_total + po.igst_total
    must_equal("total_tax", po.total_tax, taxes)
    if inter_state and (po.cgst_total or po.sgst_total):
        out.append("inter-state PO must have zero CGST/SGST totals")
    if not inter_state and po.igst_total:
        out.append("intra-state PO must have zero IGST total")

    before_rounding = (
        po.subtotal
        + taxes
        + (po.freight_charges or ZERO)
        + (po.other_charges or ZERO)
        - (po.discount_total or ZERO)
    )
    round_off = po.round_off or ZERO
    must_equal("grand_total", po.grand_total, before_rounding + round_off)
    if abs(round_off) > MAX_ROUND_OFF:
        out.append(f"round_off {round_off} exceeds {MAX_ROUND_OFF}")
    if po.grand_total != po.grand_total.quantize(RUPEE):
        out.append(f"grand_total {po.grand_total} is not rounded to whole rupees")
    if po.grand_total <= 0:
        out.append(f"grand_total {po.grand_total} must be positive")
    if po.amount_in_words is not None:
        words = amount_in_words_inr(po.grand_total)
        if po.amount_in_words != words:
            out.append(f"amount_in_words {po.amount_in_words!r} != {words!r}")
    return out
