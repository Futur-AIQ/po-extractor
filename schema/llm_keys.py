"""Short JSON keys for LLM output (PRD §6.3): fewer output tokens, mapped back in code.

The model writes {"po_no": "PO/2026-27/00457", "v_gstin": "27AAPFU0939F1ZV", ...};
to_full_keys() turns that into PurchaseOrder field names. Keys stay readable (max 8 chars),
because the model is more accurate with keys it can understand.
"""

from typing import Any

SHORT_KEY_MAX_LEN = 8

# Full PurchaseOrder field name -> short key. Prefixes: b_ buyer, v_ vendor, bt_ bill-to,
# st_ ship-to; _dt date, _no number, _stcd state code.
SHORT_KEYS: dict[str, str] = {
    # Document
    "po_number": "po_no",
    "po_date": "po_dt",
    "amendment_no": "amd_no",
    "amendment_date": "amd_dt",
    "quotation_ref": "quot_ref",
    "quotation_date": "quot_dt",
    "indent_no": "ind_no",
    "currency": "curr",
    "delivery_date": "del_dt",
    "po_validity_date": "valid_dt",
    # Buyer
    "buyer_name": "b_name",
    "buyer_address": "b_addr",
    "buyer_gstin": "b_gstin",
    "buyer_pan": "b_pan",
    "buyer_state": "b_state",
    "buyer_state_code": "b_stcd",
    "buyer_contact_person": "b_cont",
    "buyer_phone": "b_phone",
    "buyer_email": "b_email",
    # Vendor
    "vendor_name": "v_name",
    "vendor_code": "v_code",
    "vendor_address": "v_addr",
    "vendor_gstin": "v_gstin",
    "vendor_pan": "v_pan",
    "vendor_state": "v_state",
    "vendor_state_code": "v_stcd",
    "vendor_contact_person": "v_cont",
    "vendor_phone": "v_phone",
    "vendor_email": "v_email",
    # Bill-to / ship-to
    "bill_to_name": "bt_name",
    "bill_to_address": "bt_addr",
    "bill_to_gstin": "bt_gstin",
    "bill_to_state_code": "bt_stcd",
    "ship_to_name": "st_name",
    "ship_to_address": "st_addr",
    "ship_to_gstin": "st_gstin",
    "ship_to_state_code": "st_stcd",
    # Terms
    "payment_terms": "pay_term",
    "delivery_terms": "del_term",
    "freight_terms": "frt_term",
    "mode_of_transport": "transp",
    "place_of_supply": "pos",
    "delivery_location": "del_loc",
    "warranty_terms": "warranty",
    "packing_instructions": "packing",
    # Totals
    "subtotal": "subtotal",
    "discount_total": "disc_tot",
    "freight_charges": "freight",
    "other_charges": "oth_chg",
    "cgst_total": "cgst_tot",
    "sgst_total": "sgst_tot",
    "igst_total": "igst_tot",
    "total_tax": "tax_tot",  # computed, never requested; mapped for completeness
    "round_off": "rnd_off",
    "grand_total": "gr_total",
    "amount_in_words": "in_words",
    # Approval
    "prepared_by": "prep_by",
    "approved_by": "appr_by",
    # Line items (sent in a separate call as compact rows; the key only matters for dicts)
    "line_items": "items",
}

FULL_KEYS: dict[str, str] = {short: full for full, short in SHORT_KEYS.items()}


def to_short_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Rename the top-level keys of a PurchaseOrder-shaped dict to short keys."""
    return {_lookup(SHORT_KEYS, key, "field name"): value for key, value in data.items()}


def to_full_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Rename the top-level short keys of LLM output back to PurchaseOrder field names."""
    return {_lookup(FULL_KEYS, key, "short key"): value for key, value in data.items()}


def _lookup(mapping: dict[str, str], key: str, kind: str) -> str:
    if key not in mapping:
        raise KeyError(f"unknown {kind} {key!r}")
    return mapping[key]
