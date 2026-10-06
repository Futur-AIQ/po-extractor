"""JSON schemas for LLM structured output, derived from the Pydantic models.

Schemas are hand-built rather than taken from `model_json_schema()` so they stay flat and
portable: no $ref, no anyOf, only keywords that both vLLM and llama.cpp grammars accept.
All header fields are optional so the model omits absent values instead of emitting nulls.
Date fields are plain strings copied as printed; the models parse them (schema/dates.py).

v1.1 contract (PRD §6.3): header_schema_for_llm() and line_items_schema_for_llm() use short
keys and never request COMPUTED_FIELDS. The Step 1.1 functions (json_schema_for_group,
compact_line_items_schema) are kept unchanged for compatibility.
"""

import json
import types
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from schema.fields import (
    FIELD_GROUPS,
    HEADER_FIELDS_FOR_LLM,
    LINE_ITEM_COLUMNS,
    LLM_LINE_ITEM_COLUMNS,
)
from schema.llm_keys import SHORT_KEYS
from schema.po_schema import LineItem, PurchaseOrder


def _base_type(annotation: Any) -> Any:
    """Return T for `T | None` and `Annotated[T, ...]`, otherwise the annotation unchanged."""
    if get_origin(annotation) in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) != 1:
            raise TypeError(f"Unsupported union type: {annotation}")
        annotation = args[0]
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _json_type(annotation: Any) -> dict[str, Any]:
    """Map a Python field type to a simple JSON-schema type."""
    base = _base_type(annotation)
    if base is str:
        return {"type": "string"}
    if base is int:
        return {"type": "integer"}
    if base is Decimal:
        return {"type": "number"}
    if base is date:
        # No date pattern: the model copies the date as printed (a YYYY-MM-DD grammar made it
        # copy digits in print order). Parsing happens in the model validators.
        return {"type": "string"}
    raise TypeError(f"No JSON type mapping for {annotation}")


def _property(model: type[BaseModel], name: str) -> dict[str, Any]:
    """JSON-schema property for one model field, with its description."""
    field = model.model_fields[name]
    return {**_json_type(field.annotation), "description": field.description}


def json_schema_for_group(group_id: str) -> dict[str, Any]:
    """JSON schema for one header field group, for `response_format` json_schema.

    Every field is optional: the model omits fields not present in the document.
    """
    if group_id not in FIELD_GROUPS:
        raise ValueError(f"Unknown group {group_id!r}; expected one of {sorted(FIELD_GROUPS)}")
    return {
        "type": "object",
        "properties": {name: _property(PurchaseOrder, name) for name in FIELD_GROUPS[group_id]},
        "additionalProperties": False,
    }


def compact_line_items_schema() -> dict[str, Any]:
    """JSON schema for `{"rows": [[...], ...]}` with one array per line item.

    Each row lists the values in LINE_ITEM_COLUMNS order. Column names and descriptions are
    given once in the prompt, not repeated per row. Every cell may be null because array
    positions cannot be omitted; missing critical values are caught by `rows_to_line_items`.
    """
    return _rows_schema(LINE_ITEM_COLUMNS)


def _rows_schema(columns: Sequence[str], non_empty_strings: bool = False) -> dict[str, Any]:
    """`{"rows": [[...], ...]}` where each row has exactly one cell per column, in order."""
    cells = []
    for name in columns:
        cell = _json_type(LineItem.model_fields[name].annotation)
        if non_empty_strings:
            cell = _non_empty(cell)
        cell["type"] = [cell["type"], "null"]
        cells.append(cell)
    # Row length is fixed by prefixItems + min/maxItems. No `"items": false`: llama.cpp
    # rejects boolean sub-schemas ("schema must be an object").
    row = {
        "type": "array",
        "prefixItems": cells,
        "minItems": len(cells),
        "maxItems": len(cells),
    }
    return {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": row}},
        "required": ["rows"],
        "additionalProperties": False,
    }


class MalformedRowError(ValueError):
    """A compact line-item row could not be converted to a LineItem."""

    def __init__(self, row_index: int, message: str) -> None:
        self.row_index = row_index
        super().__init__(f"line-item row {row_index}: {message}")


def rows_to_line_items(
    rows: list[list[Any]], columns: Sequence[str] | None = None
) -> list[LineItem]:
    """Convert compact rows to LineItem objects.

    `columns` gives the cell order. If None, it follows the row width: 12 cells are the v1.1
    LLM_LINE_ITEM_COLUMNS (computed tax amounts left None, to be computed later), 15 cells
    are the full LINE_ITEM_COLUMNS of the Step 1.1 compact schema. Null cells are treated as
    absent. Raises MalformedRowError naming the row and the offending columns if a row has
    the wrong shape or fails LineItem validation.
    """
    items = []
    for index, row in enumerate(rows):
        if not isinstance(row, list):
            raise MalformedRowError(index, f"expected an array, got {type(row).__name__}")
        row_columns = columns if columns is not None else _columns_for_width(len(row))
        if row_columns is None or len(row) != len(row_columns):
            raise MalformedRowError(index, _width_error(columns, len(row)))
        values = {
            name: cell for name, cell in zip(row_columns, row, strict=True) if cell is not None
        }
        try:
            items.append(LineItem.model_validate(values))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            raise MalformedRowError(index, problems) from exc
    return items


_COLUMNS_BY_WIDTH = {
    len(LLM_LINE_ITEM_COLUMNS): LLM_LINE_ITEM_COLUMNS,
    len(LINE_ITEM_COLUMNS): LINE_ITEM_COLUMNS,
}


def _columns_for_width(width: int) -> list[str] | None:
    return _COLUMNS_BY_WIDTH.get(width)


def _width_error(columns: Sequence[str] | None, width: int) -> str:
    if columns is not None:
        return f"expected {len(columns)} cells, got {width}"
    return (
        f"expected {len(LINE_ITEM_COLUMNS)} cells, got {width} "
        f"({len(LLM_LINE_ITEM_COLUMNS)} also accepted: LLM_LINE_ITEM_COLUMNS)"
    )


# =========================================================================================
# v1.1 LLM output contract
# =========================================================================================


def header_schema_for_llm() -> dict[str, Any]:
    """One flat schema with every non-computed header field, keyed by short keys.

    All fields optional (absent fields are omitted, never null or ""). Descriptions are the
    model field descriptions, which name the meaning and the label synonyms. The grammar does
    not show descriptions to the model: the prompt must list the keys and their meanings.
    """
    return {
        "type": "object",
        "properties": {
            SHORT_KEYS[name]: _non_empty(_property(PurchaseOrder, name))
            for name in HEADER_FIELDS_FOR_LLM
        },
        "additionalProperties": False,
    }


def _non_empty(prop: dict[str, Any]) -> dict[str, Any]:
    """Forbid "" for strings, so the grammar itself enforces "absent fields are omitted".

    Without it a small model emitted "prepared_by": "" for fields that are not printed.
    """
    return {**prop, "minLength": 1} if prop["type"] == "string" else prop


def line_items_schema_for_llm() -> dict[str, Any]:
    """`{"rows": [[...], ...]}` with one cell per LLM_LINE_ITEM_COLUMNS column (no taxes)."""
    return _rows_schema(LLM_LINE_ITEM_COLUMNS, non_empty_strings=True)


def line_item_columns_instruction() -> str:
    """Column order for the line-item prompt: one numbered line per column with its meaning.

    Sent once in the prompt, so rows can be bare arrays instead of repeating keys per row.
    """
    lines = [
        f"Each row is an array of {len(LLM_LINE_ITEM_COLUMNS)} values in this order "
        "(null if not printed for that row):"
    ]
    for number, name in enumerate(LLM_LINE_ITEM_COLUMNS, start=1):
        lines.append(f"{number}. {name}: {LineItem.model_fields[name].description}")
    return "\n".join(lines)


def parse_llm_json(text: str) -> Any:
    """Parse LLM JSON output, reading every non-integer number as Decimal (never float)."""
    return json.loads(text, parse_float=Decimal)
