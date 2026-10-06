"""Tests for parsing dates as printed on Indian POs (schema/dates.py)."""

from datetime import date, datetime

import pytest

from schema.dates import parse_po_date
from schema.po_schema import LineItem

MARCH_14 = date(2026, 3, 14)


@pytest.mark.parametrize(
    "text",
    [
        "14/03/2026",
        "14-03-2026",
        "14.03.2026",
        "14/3/2026",
        "14/03/26",
        "  14/03/2026 ",
        "2026-03-14",
        "14-Mar-2026",
        "14-MAR-26",
        "14 Mar 2026",
        "14 March 2026",
        "14th March, 2026",
        "14.Mar.2026",
        "14 Mar. 2026",
        "March 14, 2026",
        "Mar 14th 2026",
    ],
)
def test_printed_formats_parse_day_first(text: str) -> None:
    assert parse_po_date(text) == MARCH_14


def test_ambiguous_numeric_date_is_day_first() -> None:
    assert parse_po_date("05/04/2026") == date(2026, 4, 5)  # 5 April, not May 4


def test_september_abbreviations() -> None:
    assert parse_po_date("01-Sept-2026") == parse_po_date("01-Sep-2026") == date(2026, 9, 1)


def test_date_objects_pass_through() -> None:
    assert parse_po_date(MARCH_14) == MARCH_14
    assert parse_po_date(datetime(2026, 3, 14, 10, 30)) == MARCH_14


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("03/14/2026", "invalid date"),  # month-first is rejected, never silently swapped
        ("31/02/2026", "invalid date"),
        ("14-Foo-2026", "unknown month"),
        ("1403-20-26", "invalid date"),  # the garbage the old YYYY-MM-DD grammar produced
        ("tomorrow", "unrecognised date format"),
        ("", "unrecognised date format"),
    ],
)
def test_bad_dates_raise(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_po_date(text)


def test_non_string_raises() -> None:
    with pytest.raises(ValueError, match="expected a date string"):
        parse_po_date(20260314)


def test_models_accept_printed_dates() -> None:
    item = LineItem.model_validate(
        {
            "line_no": 1,
            "description": "Bearing",
            "hsn_sac": "8482",
            "quantity": 1,
            "unit_rate": 1,
            "taxable_value": 1,
            "gst_rate": 18,
            "line_total": "1.18",
            "line_delivery_date": "15-Nov-2026",
        }
    )
    assert item.line_delivery_date == date(2026, 11, 15)
    assert item.model_dump(mode="json")["line_delivery_date"] == "2026-11-15"
