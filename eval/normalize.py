"""Per field-type normalisation, so scoring compares values, not formatting.

Every schema field (58 header fields + 15 line-item columns) has one type in FIELD_TYPES.
`normalize(field, value)` returns a comparable value, or None when the value is absent
(None, empty or blank). A value that cannot be parsed for its type (e.g. "abc" for an amount)
is kept as text, so it scores as wrong, never as absent.
"""

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from schema.dates import parse_po_date


class FieldType(StrEnum):
    IDENTIFIER = "identifier"  # GSTIN, PAN, PO number, item code, HSN: uppercase, no spaces
    NUMBER = "number"  # Decimal, compared to 2 decimal places
    DATE = "date"  # ISO yyyy-mm-dd
    EMAIL = "email"  # lowercase
    PHONE = "phone"  # digits only, last 10
    TEXT = "text"  # casefold, collapse whitespace, strip trailing punctuation


_FIELDS_BY_TYPE: dict[FieldType, list[str]] = {
    FieldType.IDENTIFIER: [
        # Header
        "po_number", "amendment_no", "quotation_ref", "indent_no", "currency",
        "buyer_gstin", "buyer_pan", "buyer_state_code",
        "vendor_code", "vendor_gstin", "vendor_pan", "vendor_state_code",
        "bill_to_gstin", "bill_to_state_code", "ship_to_gstin", "ship_to_state_code",
        # Line items
        "item_code", "hsn_sac", "uom",
    ],
    FieldType.NUMBER: [
        # Header totals
        "subtotal", "discount_total", "freight_charges", "other_charges", "cgst_total",
        "sgst_total", "igst_total", "total_tax", "round_off", "grand_total",
        # Line items
        "line_no", "quantity", "unit_rate", "discount_pct", "taxable_value", "gst_rate",
        "cgst_amount", "sgst_amount", "igst_amount", "line_total",
    ],
    FieldType.DATE: [
        "po_date", "amendment_date", "quotation_date", "delivery_date", "po_validity_date",
        "line_delivery_date",
    ],
    FieldType.EMAIL: ["buyer_email", "vendor_email"],
    FieldType.PHONE: ["buyer_phone", "vendor_phone"],
    FieldType.TEXT: [
        # Names and addresses
        "buyer_name", "buyer_address", "buyer_state", "buyer_contact_person",
        "vendor_name", "vendor_address", "vendor_state", "vendor_contact_person",
        "bill_to_name", "bill_to_address", "ship_to_name", "ship_to_address",
        # Terms and approval
        "payment_terms", "delivery_terms", "freight_terms", "mode_of_transport",
        "place_of_supply", "delivery_location", "warranty_terms", "packing_instructions",
        "amount_in_words", "prepared_by", "approved_by",
        # Line items
        "description",
    ],
}  # fmt: skip

# Field name -> type. Header and line-item names do not overlap, so one registry serves both.
FIELD_TYPES: dict[str, FieldType] = {
    field: field_type for field_type, fields in _FIELDS_BY_TYPE.items() for field in fields
}

_CENT = Decimal("0.01")
_NUMBER_NOISE = re.compile(r"[,\s₹%]|\b(?:inr|rs)\b\.?", re.IGNORECASE)
_TRAILING_PUNCTUATION = " .,;:"


def normalize_identifier(value: Any) -> str:
    return "".join(str(value).split()).upper()


def normalize_number(value: Any) -> Decimal | str:
    """'1,23,456.50', '₹ 123456.5' and 123456.5 all give Decimal('123456.50')."""
    text = _NUMBER_NOISE.sub("", str(value))
    try:
        return Decimal(text).quantize(_CENT, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return normalize_text(value)


def normalize_date(value: Any) -> str:
    try:
        return parse_po_date(value).isoformat()
    except ValueError:
        return normalize_text(value)


def normalize_email(value: Any) -> str:
    return str(value).strip().lower()


def normalize_phone(value: Any) -> str:
    """'+91 96290 10123' -> '9629010123'. Text without digits is kept as text."""
    digits = re.sub(r"\D", "", str(value))
    return digits[-10:] if digits else normalize_text(value)


def normalize_text(value: Any) -> str:
    return " ".join(str(value).casefold().split()).rstrip(_TRAILING_PUNCTUATION)


_NORMALIZERS = {
    FieldType.IDENTIFIER: normalize_identifier,
    FieldType.NUMBER: normalize_number,
    FieldType.DATE: normalize_date,
    FieldType.EMAIL: normalize_email,
    FieldType.PHONE: normalize_phone,
    FieldType.TEXT: normalize_text,
}


def is_absent(value: Any) -> bool:
    """None, empty or blank strings, and empty lists/dicts count as absent."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, list | dict) and not value


def normalize(field: str, value: Any) -> Decimal | str | None:
    """Comparable form of `value` for schema field `field`; None if the value is absent."""
    if is_absent(value):
        return None
    normalized = _NORMALIZERS[FIELD_TYPES[field]](value)
    return normalized if normalized != "" else None
