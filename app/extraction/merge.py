"""Merge: header and line-item call results -> one PO (PRD FR-09).

The header is checked field by field against the PurchaseOrder types, and rows one by one
against LineItem, so one bad value never hides the rest: every rule in validate.py checks
what is there, and the schema rule reports what is missing or invalid.

Line items (per-page mode: batches in page order):
1. Each compact row is zipped with LLM_LINE_ITEM_COLUMNS; null cells are absent values.
2. A continuation fragment (no line_no and no quantity) at the top of a page belongs to the
   previous row: its description is appended and its other values fill the gaps.
3. Rows with the same line_no (e.g. a split row given by both page calls) are deduplicated,
   keeping the most complete one.
4. Each row becomes a LineItem; failures are row errors tied to the call that produced them.

Absent CGST/SGST/IGST totals are taken as 0.00: a PO prints only the taxes that apply. This is
safe because validation compares the totals with line taxes computed in code, so a printed
tax the model missed still fails.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import TypeAdapter, ValidationError

from app.common.money import ZERO
from app.extraction.orchestrator import CallOutcome
from schema.fields import LLM_LINE_ITEM_COLUMNS
from schema.po_schema import LineItem, PurchaseOrder

TAX_TOTALS = ("cgst_total", "sgst_total", "igst_total")

# One validator per header field, with the field's own constraints (e.g. dates day-first).
_HEADER_TYPES: dict[str, TypeAdapter] = {
    name: TypeAdapter(
        Annotated[info.annotation, *info.metadata] if info.metadata else info.annotation
    )
    for name, info in PurchaseOrder.model_fields.items()
    if name != "line_items"
}


@dataclass(frozen=True)
class RowBatch:
    """The rows returned by one line-item call."""

    call_id: str  # "LI" or "LI-p{n}"
    page_no: int | None  # None: all pages (two_call)
    rows: list[Any]


@dataclass(frozen=True)
class RowError:
    """A row that could not become a LineItem."""

    call_id: str
    line_no: int | None
    message: str


@dataclass
class MergedPO:
    """A PO assembled from call results, with everything validation needs to know."""

    header: dict[str, Any]  # valid header values, by full field name
    header_errors: dict[str, str]  # field -> why its value was rejected
    items: list[LineItem]  # valid rows, in line_no order
    row_sources: dict[int, str]  # line_no -> call that produced the row
    row_errors: list[RowError]
    missing_calls: list[str] = field(default_factory=list)  # calls that returned no result
    notes: list[str] = field(default_factory=list)  # repairs and defaults applied
    supply_type: str | None = None  # "intra" | "inter", set by compute.py

    @property
    def line_calls(self) -> list[str]:
        """Line-item calls that contributed rows or errors (sorted)."""
        calls = set(self.row_sources.values()) | {e.call_id for e in self.row_errors}
        return sorted(calls | {c for c in self.missing_calls if c.startswith("LI")})


def _error_text(exc: ValidationError) -> str:
    first = exc.errors()[0]
    where = ".".join(str(part) for part in first["loc"])
    return f"{where + ': ' if where else ''}{first['msg']}"


def merge_header(raw: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """(valid values, errors by field, notes) for the header call's output."""
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, value in (raw or {}).items():
        if name not in _HEADER_TYPES:
            errors[name] = "unknown field"
            continue
        try:
            values[name] = _HEADER_TYPES[name].validate_python(value)
        except ValidationError as exc:
            errors[name] = _error_text(exc)
    notes = []
    if raw is not None:
        for name in TAX_TOTALS:
            if name not in values and name not in errors:
                values[name] = ZERO
                notes.append(f"{name} not printed: taken as 0.00")
    return values, errors, notes


def _line_key(value: Any) -> Any:
    """line_no as an int when it is one ("17", 17.0 -> 17), so duplicates are recognised."""
    try:
        number = int(str(value).strip())
    except ValueError:
        return value
    return number


def _completeness(row: dict[str, Any]) -> tuple[int, int]:
    return len(row), len(str(row.get("description", "")))


def merge_rows(
    batches: Sequence[RowBatch],
) -> tuple[list[LineItem], dict[int, str], list[RowError], list[str]]:
    """(items, row_sources, row_errors, notes) from line-item batches in page order."""
    rows: list[tuple[str, dict[str, Any]]] = []  # (call_id, values)
    errors: list[RowError] = []
    notes: list[str] = []
    for batch in batches:
        for index, cells in enumerate(batch.rows):
            if not isinstance(cells, list) or len(cells) != len(LLM_LINE_ITEM_COLUMNS):
                size = len(cells) if isinstance(cells, list) else type(cells).__name__
                message = f"expected {len(LLM_LINE_ITEM_COLUMNS)} cells, got {size}"
                errors.append(RowError(batch.call_id, None, message))
                continue
            values = {
                name: cell
                for name, cell in zip(LLM_LINE_ITEM_COLUMNS, cells, strict=True)
                if cell is not None
            }
            is_fragment = "line_no" not in values and "quantity" not in values
            if is_fragment and index == 0 and rows:
                _append_fragment(rows[-1][1], values)
                notes.append(f"{batch.call_id}: continuation of the previous row appended to it")
                continue
            rows.append((batch.call_id, values))

    kept: dict[Any, tuple[str, dict[str, Any]]] = {}  # line_no -> best row
    unnumbered: list[tuple[str, dict[str, Any]]] = []
    for call_id, values in rows:
        line_no = _line_key(values["line_no"]) if "line_no" in values else None
        if line_no is None:
            unnumbered.append((call_id, values))
        elif line_no not in kept:
            kept[line_no] = (call_id, values)
        else:
            notes.append(f"line {line_no} given twice: kept the most complete row")
            if _completeness(values) > _completeness(kept[line_no][1]):
                kept[line_no] = (call_id, values)

    items: list[LineItem] = []
    sources: dict[int, str] = {}
    for call_id, values in [*kept.values(), *unnumbered]:
        try:
            item = LineItem.model_validate(values)
        except ValidationError as exc:
            line_no = values.get("line_no") if isinstance(values.get("line_no"), int) else None
            errors.append(RowError(call_id, line_no, _error_text(exc)))
            continue
        items.append(item)
        sources[item.line_no] = call_id
    items.sort(key=lambda item: item.line_no)
    return items, sources, errors, notes


def _append_fragment(previous: dict[str, Any], fragment: dict[str, Any]) -> None:
    """Join a continuation fragment into the row it continues."""
    if fragment.get("description"):
        joined = f"{previous.get('description', '')} {fragment['description']}".strip()
        previous["description"] = joined
    for name, value in fragment.items():
        previous.setdefault(name, value)


def merge(header: dict[str, Any] | None, batches: Sequence[RowBatch]) -> MergedPO:
    """Combine the header result (full field names) and line-item batches (page order)."""
    values, header_errors, header_notes = merge_header(header)
    items, sources, row_errors, row_notes = merge_rows(batches)
    notes = header_notes + row_notes
    return MergedPO(values, header_errors, items, sources, row_errors, notes=notes)


def merge_outcomes(outcomes: Sequence[CallOutcome]) -> MergedPO:
    """Merge the orchestrator's outcomes; failed calls are listed in `missing_calls`."""
    header: dict[str, Any] | None = None
    batches: list[RowBatch] = []
    missing: list[str] = []
    for outcome in outcomes:
        spec, result = outcome.spec, outcome.result
        if not result.ok or result.parsed is None:
            missing.append(spec.call_id)
        elif spec.task.kind == "header":
            header = result.parsed
        else:
            rows = result.parsed.get("rows")
            if isinstance(rows, list):
                batches.append(RowBatch(spec.call_id, spec.task.page_no, rows))
            else:
                missing.append(spec.call_id)
    batches.sort(key=lambda b: b.page_no or 0)
    merged = merge(header, batches)
    merged.missing_calls = missing
    return merged
