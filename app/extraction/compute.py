"""Computed fields (PRD §6.3): values the model is never asked for, derived in code.

Exactly as the generator computes them (datagen/generate.py), with Decimal and ROUND_HALF_UP:
- supply type: intra-state when the vendor's state equals the place of supply, else inter;
- per line: intra -> CGST = SGST = round(taxable x rate / 2 / 100); inter -> IGST =
  round(taxable x rate / 100);
- total_tax = CGST total + SGST total + IGST total (the printed totals).

Vendor state: the GSTIN prefix, or the printed state code. Place of supply: the printed place
of supply, or else the ship-to state (state code, then GSTIN), which is where goods are
supplied under GST. If either is unknown, no line tax is computed and validation fails the
tax-regime rule, so the PO is retried or reviewed instead of guessed.
"""

from dataclasses import replace
from typing import Any, Literal

from app.common.gst import STATES, is_valid_gstin, state_code_from_place
from app.common.money import ZERO, round_money
from app.extraction.merge import TAX_TOTALS, MergedPO

SupplyType = Literal["intra", "inter"]


def vendor_state_code(header: dict[str, Any]) -> str | None:
    """The vendor's GST state code: GSTIN prefix, else the printed state code."""
    gstin = header.get("vendor_gstin")
    if gstin and is_valid_gstin(gstin):
        return gstin[:2]
    code = header.get("vendor_state_code")
    return code if code in STATES else None


def place_of_supply_code(header: dict[str, Any]) -> str | None:
    """The place-of-supply state code: printed place of supply, else the ship-to state."""
    if code := state_code_from_place(header.get("place_of_supply")):
        return code
    if header.get("ship_to_state_code") in STATES:
        return header["ship_to_state_code"]
    gstin = header.get("ship_to_gstin")
    return gstin[:2] if gstin and is_valid_gstin(gstin) else None


def supply_type(header: dict[str, Any]) -> SupplyType | None:
    """Intra- or inter-state supply, or None when either state is unknown."""
    vendor, place = vendor_state_code(header), place_of_supply_code(header)
    if vendor is None or place is None:
        return None
    return "intra" if vendor == place else "inter"


def compute(merged: MergedPO) -> MergedPO:
    """A copy of `merged` with line taxes, total_tax and supply_type filled in."""
    kind = supply_type(merged.header)
    items = []
    for item in merged.items:
        if kind == "inter":
            igst = round_money(item.taxable_value * item.gst_rate / 100)
            taxes = {"cgst_amount": None, "sgst_amount": None, "igst_amount": igst}
            item = item.model_copy(update=taxes)
        elif kind == "intra":
            half = round_money(item.taxable_value * item.gst_rate / 2 / 100)
            taxes = {"cgst_amount": half, "sgst_amount": half, "igst_amount": None}
            item = item.model_copy(update=taxes)
        items.append(item)
    header = dict(merged.header)
    if all(name in header for name in TAX_TOTALS):
        header["total_tax"] = sum((header[name] for name in TAX_TOTALS), ZERO)
    return replace(merged, header=header, items=items, supply_type=kind)
