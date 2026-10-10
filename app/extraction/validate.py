"""Validation engine: every rule of PRD §9.4 as one function returning a CheckResult.

Each rule checks what the merged PO has; a missing or invalid value is reported once, by the
schema rule, and the other rules skip what they cannot check. A failed check names its
retry target, the call to re-run in Step 3.7:

    "H"        the header call
    "LI-p{n}"  one line-item call (or "LI" in two_call mode)
    "LI"       every line-item call (the problem spans several)
    "both"     the header call and every line-item call
    None       nothing to re-run (a duplicate PO goes straight to review)

Line-item fields are named "line_items[<line_no>].<field>" for highlighting in the dashboard.
Money is compared with Decimal, within MONEY_TOLERANCE (PRD rule 4: +/-1.00).
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from app.common.amount_words import parse_amount_in_words
from app.common.gst import STATES, is_valid_gstin, is_valid_pan, state_code_from_place
from app.common.money import ZERO, round_money
from app.extraction.grounding import GroundingIndex, normalise
from app.extraction.merge import MergedPO
from schema.fields import CRITICAL_HEADER_FIELDS, HEADER_FIELDS
from schema.po_schema import PurchaseOrder

MONEY_TOLERANCE = Decimal("1.00")
MAX_ROUND_OFF = Decimal("1.00")
HSN_SAC_FORMAT = re.compile(r"[0-9]{4,8}")
# ISO 4217 codes an Indian PO plausibly uses.
KNOWN_CURRENCIES = frozenset(
    {"INR", "USD", "EUR", "GBP", "JPY", "CNY", "AED", "SGD", "CHF", "AUD", "CAD", "HKD", "SAR"}
)
PARTIES = ("buyer", "vendor", "bill_to", "ship_to")
MAX_DETAIL_ITEMS = 5  # problems listed in a check's detail text


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one validation rule."""

    rule: str
    passed: bool
    fields: tuple[str, ...] = ()  # fields to highlight
    detail: str = ""
    retry_target: str | None = None  # see module docstring
    retryable: bool = True


@dataclass(frozen=True)
class ValidationReport:
    """All checks of one PO."""

    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]


@dataclass(frozen=True)
class Masters:
    """Mock ERP master data (rule 10), indexed by PAN: all units of a company share it."""

    party_gstins: frozenset[str]
    item_codes_by_pan: dict[str, frozenset[str]]
    processed_by_pan: dict[str, frozenset[str]]  # normalised PO numbers


@lru_cache
def load_masters(path: Path) -> Masters:
    """Load parties.json, items.json and processed_po_numbers.json from `path` (once)."""
    parties = json.loads((path / "parties.json").read_text(encoding="utf-8"))
    items = json.loads((path / "items.json").read_text(encoding="utf-8"))
    processed = json.loads((path / "processed_po_numbers.json").read_text(encoding="utf-8"))
    codes: dict[str, set[str]] = {}
    for item in items:
        codes.setdefault(item["issuer_gstin"][2:12], set()).add(item["item_code"])
    numbers: dict[str, set[str]] = {}
    for gstin, entry in processed.items():
        numbers.setdefault(gstin[2:12], set()).update(normalise(n) for n in entry["po_numbers"])
    return Masters(
        party_gstins=frozenset(party["gstin"] for party in parties),
        item_codes_by_pan={pan: frozenset(c) for pan, c in codes.items()},
        processed_by_pan={pan: frozenset(n) for pan, n in numbers.items()},
    )


# =========================================================================================
# Helpers
# =========================================================================================


def _line_field(line_no: int | None, name: str = "") -> str:
    where = f"line_items[{line_no if line_no is not None else '?'}]"
    return f"{where}.{name}" if name else where


def _target(header: bool, line_calls: Iterable[str]) -> str | None:
    """The retry target for problems in the header and/or rows from `line_calls`."""
    calls = set(line_calls)
    if header and calls:
        return "both"
    if header:
        return "H"
    if len(calls) == 1:
        return calls.pop()
    return "LI" if calls else None


def _calls_of(merged: MergedPO, line_nos: Iterable[int]) -> set[str]:
    return {merged.row_sources[n] for n in line_nos if n in merged.row_sources}


def _summary(problems: list[str]) -> str:
    shown = "; ".join(problems[:MAX_DETAIL_ITEMS])
    more = len(problems) - MAX_DETAIL_ITEMS
    return f"{shown}; and {more} more" if more > 0 else shown


def _passed(rule: str, detail: str = "") -> CheckResult:
    return CheckResult(rule, True, detail=detail)


def _differs(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) > MONEY_TOLERANCE


# =========================================================================================
# Rules
# =========================================================================================


def check_schema(merged: MergedPO) -> CheckResult:
    """Rule 1: values have valid types and every critical field is present."""
    h = merged.header
    problems = [f"{name}: {error}" for name, error in merged.header_errors.items()]
    missing = sorted(CRITICAL_HEADER_FIELDS - h.keys() - merged.header_errors.keys())
    problems += [f"{name}: missing" for name in missing]
    fields = [*merged.header_errors, *missing]
    for error in merged.row_errors:
        problems.append(f"{error.call_id} {_line_field(error.line_no)}: {error.message}")
        fields.append(_line_field(error.line_no))
    problems += [f"{call}: no result" for call in merged.missing_calls]
    header = bool(merged.header_errors or missing or "H" in merged.missing_calls)
    line_calls = {e.call_id for e in merged.row_errors}
    line_calls |= {c for c in merged.missing_calls if c != "H"}
    if not problems:
        return _passed("schema")
    return CheckResult(
        "schema", False, tuple(fields), _summary(problems), _target(header, line_calls)
    )


def check_gstin(merged: MergedPO) -> CheckResult:
    """Rule 2: GSTIN format and checksum; its state code matches the state fields; the PAN is
    valid and equals GSTIN characters 3-12."""
    h = merged.header
    problems, fields = [], []
    for party in PARTIES:
        gstin = h.get(f"{party}_gstin")
        code = h.get(f"{party}_state_code")
        state = h.get(f"{party}_state")
        pan = h.get(f"{party}_pan")
        if gstin is not None and not is_valid_gstin(gstin):
            problems.append(f"{party}_gstin {gstin} fails the format/checksum")
            fields.append(f"{party}_gstin")
        valid = gstin is not None and is_valid_gstin(gstin)
        if code is not None and (code not in STATES or (valid and code != gstin[:2])):
            problems.append(f"{party}_state_code {code} does not match GSTIN {gstin}")
            fields += [f"{party}_state_code", f"{party}_gstin"]
        named = state_code_from_place(state)
        if valid and named is not None and named != gstin[:2]:
            problems.append(f"{party}_state {state!r} does not match GSTIN state {gstin[:2]}")
            fields += [f"{party}_state", f"{party}_gstin"]
        if pan is not None and not is_valid_pan(pan):
            problems.append(f"{party}_pan {pan} is not a valid PAN")
            fields.append(f"{party}_pan")
        elif pan is not None and valid and pan != gstin[2:12]:
            problems.append(f"{party}_pan {pan} != PAN in GSTIN {gstin[2:12]}")
            fields += [f"{party}_pan", f"{party}_gstin"]
    if not problems:
        return _passed("gstin")
    return CheckResult("gstin", False, tuple(dict.fromkeys(fields)), _summary(problems), "H")


def check_tax_regime(merged: MergedPO) -> CheckResult:
    """Rule 3: intra-state -> CGST + SGST, IGST 0; inter-state -> IGST only."""
    h = merged.header
    if merged.supply_type is None:
        detail = "cannot tell intra- from inter-state: vendor state or place of supply unknown"
        fields = ("vendor_gstin", "place_of_supply")
        return CheckResult("tax_regime", False, fields, detail, "H")
    if merged.supply_type == "intra":
        wrong = [name for name in ("igst_total",) if h.get(name, ZERO) != 0]
    else:
        wrong = [name for name in ("cgst_total", "sgst_total") if h.get(name, ZERO) != 0]
    if not wrong:
        return _passed("tax_regime", f"{merged.supply_type}-state supply")
    detail = f"{merged.supply_type}-state supply but {', '.join(wrong)} is not 0"
    return CheckResult("tax_regime", False, (*wrong, "place_of_supply"), detail, "H")


def check_line_arithmetic(merged: MergedPO) -> CheckResult:
    """Rule 4: taxable = qty x rate x (1 - discount%); line_total = taxable + line taxes."""
    problems, fields, bad_lines = [], [], []
    for item in merged.items:
        n = item.line_no
        discount = item.discount_pct or ZERO
        expected = round_money(item.quantity * item.unit_rate * (1 - discount / 100))
        if _differs(item.taxable_value, expected):
            problems.append(f"line {n}: taxable {item.taxable_value} != qty x rate = {expected}")
            fields += [_line_field(n, name) for name in ("taxable_value", "quantity", "unit_rate")]
            bad_lines.append(n)
        if merged.supply_type is not None:
            taxes = (item.cgst_amount or ZERO) + (item.sgst_amount or ZERO)
            taxes += item.igst_amount or ZERO
            if _differs(item.line_total, item.taxable_value + taxes):
                total = item.taxable_value + taxes
                problems.append(f"line {n}: line_total {item.line_total} != {total}")
                fields += [_line_field(n, "line_total"), _line_field(n, "gst_rate")]
                bad_lines.append(n)
    if not problems:
        return _passed("line_arithmetic")
    target = _target(False, _calls_of(merged, bad_lines))
    return CheckResult("line_arithmetic", False, tuple(fields), _summary(problems), target)


def check_totals(merged: MergedPO) -> CheckResult:
    """Rule 5: totals agree with the lines; grand total = subtotal + taxes + charges - discount
    + round off; |round off| <= 1.

    Retry target: "both" when the lines disagree with the totals; only the line calls when
    rows are visibly incomplete (that explains the difference); "H" for header-only sums.
    """
    h, items = merged.header, merged.items
    lines_vs_header, header_only, fields = [], [], []
    if items and "subtotal" in h:
        taxable = sum((item.taxable_value for item in items), ZERO)
        if _differs(taxable, h["subtotal"]):
            lines_vs_header.append(f"sum of taxable values {taxable} != subtotal {h['subtotal']}")
            fields.append("subtotal")
    if items and merged.supply_type is not None:
        for tax in ("cgst", "sgst", "igst"):
            if f"{tax}_total" not in h:
                continue
            computed = sum((getattr(item, f"{tax}_amount") or ZERO for item in items), ZERO)
            if _differs(computed, h[f"{tax}_total"]):
                printed = h[f"{tax}_total"]
                lines_vs_header.append(f"{tax} of the lines {computed} != {tax}_total {printed}")
                fields.append(f"{tax}_total")
    needed = ("subtotal", "cgst_total", "sgst_total", "igst_total", "grand_total")
    if all(name in h for name in needed):
        taxes = h["cgst_total"] + h["sgst_total"] + h["igst_total"]
        charges = h.get("freight_charges", ZERO) + h.get("other_charges", ZERO)
        adjustments = h.get("round_off", ZERO) - h.get("discount_total", ZERO)
        expected = h["subtotal"] + taxes + charges + adjustments
        if _differs(expected, h["grand_total"]):
            header_only.append(f"grand_total {h['grand_total']} != components {expected}")
            fields.append("grand_total")
    if abs(h.get("round_off", ZERO)) > MAX_ROUND_OFF:
        header_only.append(f"round_off {h['round_off']} exceeds {MAX_ROUND_OFF}")
        fields.append("round_off")
    if not lines_vs_header and not header_only:
        return _passed("totals")
    # Lines that disagree with the totals point at both calls, unless rows are visibly
    # incomplete (a gap, a bad row): then the missing rows explain it and their calls are
    # re-run. If the header was wrong too, the PO still fails after the retry: review.
    rows_target = _incomplete_rows_target(merged) if lines_vs_header else None
    if rows_target and not header_only:
        target = rows_target
    elif lines_vs_header:
        target = "both"
    else:
        target = "H"
    detail = _summary(lines_vs_header + header_only)
    return CheckResult("totals", False, tuple(dict.fromkeys(fields)), detail, target)


def _gaps(merged: MergedPO) -> tuple[list[int], int, set[str]]:
    """(missing line numbers, step, calls that produced the rows around each gap)."""
    numbers = [item.line_no for item in merged.items]
    if not numbers:
        return [], 1, set()
    step = 10 if all(n % 10 == 0 for n in numbers) else 1
    present = set(numbers)
    missing = [n for n in range(step, max(numbers) + 1, step) if n not in present]
    neighbours = set()
    for gap in missing:
        before = [n for n in numbers if n < gap]
        after = [n for n in numbers if n > gap]
        neighbours |= {max(before)} if before else set()
        neighbours |= {min(after)} if after else set()
    return missing, step, _calls_of(merged, neighbours)


def _incomplete_rows_target(merged: MergedPO) -> str | None:
    """The retry target if rows are visibly incomplete (gap, bad row, failed line call)."""
    _, _, gap_calls = _gaps(merged)
    calls = gap_calls | {e.call_id for e in merged.row_errors}
    calls |= {c for c in merged.missing_calls if c != "H"}
    return _target(False, calls)


def check_continuity(merged: MergedPO) -> CheckResult:
    """Rule 6: line numbers run 1, 2, 3 ... (or 10, 20, 30 ...) with no gaps."""
    if not merged.items:
        return CheckResult("continuity", False, ("line_items",), "no line items", "LI")
    missing, step, calls = _gaps(merged)
    if not missing:
        return _passed("continuity", f"{len(merged.items)} rows, step {step}")
    detail = f"missing line numbers {missing[: MAX_DETAIL_ITEMS * 4]}"
    fields = tuple(_line_field(n) for n in missing)
    return CheckResult("continuity", False, fields, detail, _target(False, calls))


def check_amount_in_words(merged: MergedPO) -> CheckResult:
    """Rule 7: the amount in words reads as the grand total (paise may be left out)."""
    h = merged.header
    words, total = h.get("amount_in_words"), h.get("grand_total")
    if words is None or total is None:
        return _passed("amount_in_words", "not printed" if words is None else "")
    try:
        amount = parse_amount_in_words(words)
    except ValueError as exc:
        return CheckResult("amount_in_words", False, ("amount_in_words",), str(exc), "H")
    if amount == total or (amount == amount.to_integral_value() and amount == int(total)):
        return _passed("amount_in_words")
    detail = f"words say {amount}, grand_total is {total}"
    return CheckResult("amount_in_words", False, ("amount_in_words", "grand_total"), detail, "H")


def check_formats(merged: MergedPO) -> CheckResult:
    """Rule 8: dates consistent with the PO date, numeric HSN/SAC (4-8 digits), known currency."""
    h = merged.header
    po_date: date | None = h.get("po_date")
    header_problems, row_problems, fields, bad_lines = [], [], [], []
    if po_date is not None:
        for name in ("delivery_date", "po_validity_date", "amendment_date"):
            if h.get(name) is not None and h[name] < po_date:
                header_problems.append(f"{name} {h[name]} is before po_date {po_date}")
                fields.append(name)
        if h.get("quotation_date") is not None and h["quotation_date"] > po_date:
            header_problems.append(f"quotation_date {h['quotation_date']} is after po_date")
            fields.append("quotation_date")
    currency = h.get("currency")
    if currency is not None and currency.upper() not in KNOWN_CURRENCIES:
        header_problems.append(f"unknown currency {currency!r}")
        fields.append("currency")
    for item in merged.items:
        n = item.line_no
        if not HSN_SAC_FORMAT.fullmatch(item.hsn_sac):
            row_problems.append(f"line {n}: HSN/SAC {item.hsn_sac!r} is not 4-8 digits")
            fields.append(_line_field(n, "hsn_sac"))
            bad_lines.append(n)
        if po_date and item.line_delivery_date and item.line_delivery_date < po_date:
            row_problems.append(f"line {n}: delivery date {item.line_delivery_date} before PO")
            fields.append(_line_field(n, "line_delivery_date"))
            bad_lines.append(n)
    if not header_problems and not row_problems:
        return _passed("formats")
    target = _target(bool(header_problems), _calls_of(merged, bad_lines))
    detail = _summary(header_problems + row_problems)
    return CheckResult("formats", False, tuple(fields), detail, target)


GROUNDED_HEADER_FIELDS = ("po_number", *(f"{p}_gstin" for p in PARTIES), "buyer_pan", "vendor_pan")


def check_grounding(merged: MergedPO, index: GroundingIndex | None) -> CheckResult:
    """Rule 9: identifiers appear in the native text layer. "unverifiable" (the value may be on
    a page without text) passes with a note; only "no" fails."""
    if index is None:
        return _passed("grounding", "no text layer index")
    h = merged.header
    values = [(name, None, h[name]) for name in GROUNDED_HEADER_FIELDS if h.get(name)]
    for item in merged.items:
        for name in ("item_code", "hsn_sac"):
            if getattr(item, name):
                values.append((name, item.line_no, getattr(item, name)))
    not_found, fields, bad_lines, unverifiable = [], [], [], 0
    header = False
    for name, line_no, value in values:
        verdict = index.is_grounded(str(value))
        if verdict == "unverifiable":
            unverifiable += 1
        elif verdict == "no":
            where = name if line_no is None else _line_field(line_no, name)
            not_found.append(f"{where} {value!r} not in the text layer")
            fields.append(where)
            if line_no is None:
                header = True
            else:
                bad_lines.append(line_no)
    note = f"{unverifiable} of {len(values)} identifiers unverifiable (no text layer)"
    if not not_found:
        return _passed("grounding", note if unverifiable else "")
    target = _target(header, _calls_of(merged, bad_lines))
    return CheckResult("grounding", False, tuple(fields), _summary(not_found), target)


def check_masters(merged: MergedPO, masters: Masters | None) -> CheckResult:
    """Rule 10: buyer and vendor GSTINs are known parties; item codes are in the buyer's item
    master."""
    if masters is None:
        return _passed("masters", "no master data")
    h = merged.header
    problems, fields, bad_lines, notes = [], [], [], []
    for name in ("buyer_gstin", "vendor_gstin"):
        if h.get(name) and h[name] not in masters.party_gstins:
            problems.append(f"{name} {h[name]} is not in the party master")
            fields.append(name)
    header = bool(problems)
    buyer = h.get("buyer_gstin")
    known = masters.item_codes_by_pan.get(buyer[2:12]) if buyer else None
    if known is None:
        notes.append("no item master for the buyer")
    else:
        for item in merged.items:
            if item.item_code and item.item_code not in known:
                problems.append(f"line {item.line_no}: item code {item.item_code} unknown")
                fields.append(_line_field(item.line_no, "item_code"))
                bad_lines.append(item.line_no)
    if not problems:
        return _passed("masters", "; ".join(notes))
    target = _target(header, _calls_of(merged, bad_lines))
    return CheckResult("masters", False, tuple(fields), _summary(problems), target)


def check_duplicate_po(merged: MergedPO, masters: Masters | None) -> CheckResult:
    """Rule 10: the PO number was not processed before for this buyer. Not retryable: a
    duplicate goes straight to review."""
    h = merged.header
    if masters is None or not h.get("po_number") or not h.get("buyer_gstin"):
        return _passed("duplicate_po", "not checked")
    processed = masters.processed_by_pan.get(h["buyer_gstin"][2:12], frozenset())
    if normalise(h["po_number"]) not in processed:
        return _passed("duplicate_po")
    detail = f"PO {h['po_number']} was already processed for buyer {h['buyer_gstin']}"
    return CheckResult("duplicate_po", False, ("po_number",), detail, None, retryable=False)


def validate(
    merged: MergedPO, grounding: GroundingIndex | None = None, masters: Masters | None = None
) -> ValidationReport:
    """Run every rule (merged must already have computed fields: compute.compute)."""
    return ValidationReport(
        [
            check_schema(merged),
            check_gstin(merged),
            check_tax_regime(merged),
            check_line_arithmetic(merged),
            check_totals(merged),
            check_continuity(merged),
            check_amount_in_words(merged),
            check_formats(merged),
            check_grounding(merged, grounding),
            check_masters(merged, masters),
            check_duplicate_po(merged, masters),
        ]
    )


def to_purchase_order(merged: MergedPO) -> PurchaseOrder | None:
    """The merged PO as a PurchaseOrder, or None while required values are missing."""
    header = {name: merged.header[name] for name in HEADER_FIELDS if name in merged.header}
    try:
        return PurchaseOrder.model_validate({**header, "line_items": merged.items})
    except ValidationError:
        return None
