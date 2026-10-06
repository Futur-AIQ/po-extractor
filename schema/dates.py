"""Deterministic parsing of dates as printed on Indian purchase orders.

The LLM copies dates exactly as printed; this module turns them into `datetime.date`.
Asking the model to reformat dates itself proved unreliable: under a YYYY-MM-DD grammar a
small model copied digits in print order ("14/03/2026" -> "1403-20-26").

Numeric dates are read day-first (DD/MM/YYYY, the Indian convention). A date that is only
valid month-first, e.g. "03/14/2026", is rejected, never silently swapped.
"""

import re
from datetime import date, datetime
from typing import Annotated, Any

from pydantic import BeforeValidator

_MONTHS = {
    name: number
    for number, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}

_SUFFIX = r"(?:st|nd|rd|th)?"
# 2026-03-14 (ISO, also what JSON ground truth contains)
_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
# 14/03/2026, 14-03-2026, 14.03.26
_DAY_FIRST = re.compile(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})")
# 14-Mar-2026, 14 March 2026, 14th Mar, 2026, 14.Mar.26
_DAY_MONTH_NAME = re.compile(rf"(\d{{1,2}}){_SUFFIX}[ /.-]*([a-z]+)\.?,?[ /.-]*(\d{{4}}|\d{{2}})")
# March 14, 2026 / Mar 14th 2026
_MONTH_NAME_DAY = re.compile(rf"([a-z]+)\.? (\d{{1,2}}){_SUFFIX},? (\d{{4}})")


def _year(text: str) -> int:
    """Four-digit year; two-digit years are taken as 20YY."""
    return int(text) if len(text) == 4 else 2000 + int(text)


def _month(name: str) -> int:
    if name not in _MONTHS:
        raise ValueError(f"unknown month name {name!r}")
    return _MONTHS[name]


def parse_po_date(value: Any) -> date:
    """Parse a date as printed on a PO (or an ISO string / date object) into a `date`.

    Raises ValueError for anything unrecognised or impossible (e.g. 31/02/2026).
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"expected a date string, got {type(value).__name__}")

    text = " ".join(value.strip().lower().split())
    try:
        if m := _ISO.fullmatch(text):
            return date(int(m[1]), int(m[2]), int(m[3]))
        if m := _DAY_FIRST.fullmatch(text):
            return date(_year(m[3]), int(m[2]), int(m[1]))
        if m := _DAY_MONTH_NAME.fullmatch(text):
            return date(_year(m[3]), _month(m[2]), int(m[1]))
        if m := _MONTH_NAME_DAY.fullmatch(text):
            return date(int(m[3]), _month(m[1]), int(m[2]))
    except ValueError as exc:  # e.g. day 31 in February, month 14
        raise ValueError(f"invalid date {value!r}: {exc}") from exc
    raise ValueError(f"unrecognised date format {value!r}; expected e.g. 14/03/2026")


# Date field type for the PO models: accepts printed dates, stores a real `date`.
PODate = Annotated[date, BeforeValidator(parse_po_date)]
