"""Regex hints from native text layers (PRD FR-06, §9.4 rules 2, 8, 9).

Deterministic candidates for identifiers the LLM extracts: GSTINs, PANs, emails, phones,
dates and HSN/SAC codes. Only text layers of native pages are read; scanned pages have none.

Hints are used for validation and grounding only. They are NOT injected into the prompt:
- the prompt prefix stays identical for every PO, so prefix caching keeps working;
- the model reads the page itself instead of copying a candidate list, so a wrong or
  ambiguous hint cannot bias it (a hint does not say which party a GSTIN belongs to);
- hints stay an independent check on the model's output instead of an input to it.

The lists are candidates, not answers: they favour recall. For example, a 10-digit
document number on its own line looks like a phone, and an 8-digit item code looks like
an HSN code. Each list keeps the order of first appearance, without duplicates.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.common.gst import is_valid_gstin, is_valid_pan
from app.extraction.preprocess import PreparedDoc
from schema.dates import parse_po_date


@dataclass(frozen=True)
class RegexHints:
    """Identifiers found in a PO's native text layers."""

    gstins: tuple[str, ...] = ()  # valid format, state code and checksum
    invalid_gstins: tuple[str, ...] = ()  # GSTIN-shaped, but bad state code or checksum
    pans: tuple[str, ...] = ()  # printed on their own (not the PAN inside a GSTIN)
    emails: tuple[str, ...] = ()  # lowercase
    phones: tuple[str, ...] = ()  # digits only; a leading "+" is kept ("+919820012345")
    dates: tuple[str, ...] = ()  # ISO yyyy-mm-dd, read day-first as printed in India
    hsn_codes: tuple[str, ...] = ()  # HSN/SAC-like codes


# GSTIN shape with any 3 last characters, so typos in the "Z" or check position are caught.
_GSTIN_LIKE = re.compile(r"(?<![A-Z0-9])[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]{3}(?![A-Z0-9])", re.I)
_PAN_LIKE = re.compile(r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])", re.I)
_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)*\.[A-Z]{2,}", re.I)

# Indian numbers: an optional +91 or trunk 0, then 10 digits as 98200 12345, 982 001 2345,
# 9820012345, or a landline as 022-23456789. Not part of a longer number or code.
_PHONE = re.compile(
    r"(?<![\w.,/+-])(?:\+91[ -]?|0)?"
    r"(?:[0-9]{10}|[0-9]{5}[ -][0-9]{5}|[0-9]{3}[ -][0-9]{3}[ -][0-9]{4}|0[0-9]{2,4}[ -][0-9]{6,8})"
    r"(?![\w,/-]|\.[0-9])"
)
_PHONE_LABEL = re.compile(
    r"\b(?:ph(?:one)?|tel(?:ephone)?|mob(?:ile)?|cell|contact(?:\s*no)?)\b[^\n]{0,3}\W*$", re.I
)
_PHONE_LABEL_WINDOW = 25  # characters before the number searched for a label

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = re.compile(
    r"(?<![0-9])(?<![0-9][/.-])(?:"  # not inside a longer number
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}"  # 2026-03-14
    r"|[0-9]{1,2}[/.-][0-9]{1,2}[/.-](?:[0-9]{4}|[0-9]{2})"  # 14/03/2026, 14.03.26
    rf"|[0-9]{{1,2}}(?:st|nd|rd|th)?[ /.-]*{_MONTH},?[ /.-]*[0-9]{{4}}"  # 14-Mar-2026, 14th March
    rf"|\b{_MONTH} [0-9]{{1,2}}(?:st|nd|rd|th)?,? [0-9]{{4}}"  # March 14, 2026
    r")(?![0-9]|[/.-][0-9])",
    re.I,
)

# HSN codes have 4, 6 or 8 digits; SAC codes 6 digits starting with 99. Six and eight digit
# numbers that stand alone are candidates; four-digit ones only after an HSN/SAC label, as
# descriptions are full of them (1440 rpm, IS 2062). PIN codes ("Pune - 411001") are excluded.
_HSN_STANDALONE = re.compile(
    r"(?<![\w,./:#-])(?<!- )(?<!PIN )([0-9]{6}|[0-9]{8})(?![\w,/:-]|\.[0-9])", re.I
)
_HSN_LABELLED = re.compile(
    r"\b(?:HSN|SAC)(?:/SAC)?(?:\s*(?:code|no\.?))?\s*[:-]?\s*([0-9]{4}|[0-9]{6}|[0-9]{8})\b", re.I
)
_INVALID_HSN_CHAPTERS = {"00", "77"}  # chapter 77 is reserved in the HSN


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _gstins(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidates = _unique(match.upper() for match in _GSTIN_LIKE.findall(text))
    valid = tuple(g for g in candidates if is_valid_gstin(g))
    return valid, tuple(g for g in candidates if g not in valid)


def _phones(text: str) -> tuple[str, ...]:
    """Numbers with a +91 prefix, a phone label just before them, or a line of their own."""
    found = []
    for match in _PHONE.finditer(text):
        number = match.group(0)
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        line = text[line_start : line_end if line_end != -1 else len(text)]
        before = text[max(0, match.start() - _PHONE_LABEL_WINDOW) : match.start()]
        if number.startswith("+91") or _PHONE_LABEL.search(before) or line.strip() == number:
            found.append(("+" if number.startswith("+") else "") + re.sub(r"\D", "", number))
    return _unique(found)


def _dates(text: str) -> tuple[str, ...]:
    found = []
    for match in _DATE.finditer(text):
        try:
            found.append(parse_po_date(match.group(0)).isoformat())
        except ValueError:  # 31/02/2026, month-first 03/14/2026, "14 Mayor 2026"
            continue
    return _unique(found)


def _hsn_codes(text: str) -> tuple[str, ...]:
    codes = _HSN_STANDALONE.findall(text) + _HSN_LABELLED.findall(text)
    return _unique(code for code in codes if code[:2] not in _INVALID_HSN_CHAPTERS)


def extract_hints(texts: Iterable[str]) -> RegexHints:
    """Regex hints from page text layers (one string per page)."""
    text = "\n".join(texts)
    gstins, invalid_gstins = _gstins(text)
    return RegexHints(
        gstins=gstins,
        invalid_gstins=invalid_gstins,
        pans=_unique(p.upper() for p in _PAN_LIKE.findall(text) if is_valid_pan(p.upper())),
        emails=_unique(e.lower() for e in _EMAIL.findall(text)),
        phones=_phones(text),
        dates=_dates(text),
        hsn_codes=_hsn_codes(text),
    )


def hints_from_doc(doc: PreparedDoc) -> RegexHints:
    """Regex hints from the text layers of a prepared PO (native pages sent to the LLM)."""
    return extract_hints(page.text_md for page in doc.kept_pages if page.text_md)
