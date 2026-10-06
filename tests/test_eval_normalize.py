"""Tests for per field-type normalisation (Step 2.1)."""

import types
from datetime import date
from decimal import Decimal
from typing import Annotated, Union, get_args, get_origin

import pytest

from eval.normalize import _FIELDS_BY_TYPE, FIELD_TYPES, FieldType, normalize
from schema.fields import HEADER_FIELDS, LINE_ITEM_COLUMNS
from schema.po_schema import LineItem, PurchaseOrder


def _base_types(annotation: object) -> set[object]:
    """`Decimal | None` -> {Decimal}; `PODate | None` -> {date}."""
    if get_origin(annotation) in (Union, types.UnionType):
        return set().union(*(_base_types(a) for a in get_args(annotation) if a is not type(None)))
    if get_origin(annotation) is Annotated:
        return {get_args(annotation)[0]}
    return {annotation}


def test_every_schema_field_has_exactly_one_type() -> None:
    listed = [name for names in _FIELDS_BY_TYPE.values() for name in names]
    assert len(listed) == len(set(listed)), "a field is listed under two types"
    assert set(FIELD_TYPES) == set(HEADER_FIELDS) | set(LINE_ITEM_COLUMNS)


def test_registry_agrees_with_schema_types() -> None:
    fields = PurchaseOrder.model_fields | LineItem.model_fields
    for name, field_type in FIELD_TYPES.items():
        base = _base_types(fields[name].annotation)
        if base & {Decimal, int}:
            assert field_type is FieldType.NUMBER, name
        elif date in base:
            assert field_type is FieldType.DATE, name
        else:
            assert field_type not in (FieldType.NUMBER, FieldType.DATE), name


@pytest.mark.parametrize(
    ("field", "a", "b"),
    [
        ("grand_total", "1,23,456.50", 123456.5),  # Indian grouping vs float
        ("grand_total", "₹ 1,23,456.5", "123456.50"),
        ("grand_total", "Rs. 500", Decimal("500.004")),  # compared to 2 decimal places
        ("gst_rate", "18%", 18),
        ("line_no", "7", 7),
        ("vendor_gstin", "27 aapcs 1234 k1z5", "27AAPCS1234K1Z5"),
        ("hsn_sac", 271019, "2710 19"),
        ("po_date", "13/09/2025", "2025-09-13"),
        ("po_date", "13-Sep-2025", date(2025, 9, 13)),
        ("buyer_email", " Purchase@ACME.com ", "purchase@acme.com"),
        ("vendor_phone", "+91 96290 10123", "9629010123"),
        ("vendor_phone", "096290-10123", "9629010123"),
        ("buyer_name", "Acme  Steels Pvt. Ltd.", "ACME STEELS PVT. LTD"),
        ("buyer_address", "Plot 4,\nMIDC Area,", "plot 4, midc area"),
    ],
)
def test_equal_after_normalisation(field: str, a: object, b: object) -> None:
    assert normalize(field, a) == normalize(field, b)


@pytest.mark.parametrize(
    ("field", "a", "b"),
    [
        ("grand_total", "123456.50", "123456.51"),
        ("vendor_gstin", "27AAPCS1234K1Z5", "27AAPCS1234K1Z6"),
        ("po_date", "13/09/2025", "09/13/2025"),  # month-first is invalid, not swapped
        ("buyer_name", "Acme Steels", "Acme Steel"),
    ],
)
def test_different_after_normalisation(field: str, a: object, b: object) -> None:
    assert normalize(field, a) != normalize(field, b)


def test_absent_values_and_unparseable_values() -> None:
    for value in (None, "", "   ", []):
        assert normalize("payment_terms", value) is None
    assert normalize("prepared_by", "...") is None  # only punctuation
    assert normalize("grand_total", "abc") == "abc"  # kept: scores as wrong, not absent
    assert normalize("po_date", "sometime") == "sometime"
