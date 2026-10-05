"""JSON schemas for LLM structured output, derived from the Pydantic models.

Schemas are hand-built rather than taken from `model_json_schema()` so they stay flat and
portable: no $ref, no anyOf, only keywords that both vLLM and llama.cpp grammars accept.
All header fields are optional so the model omits absent values instead of emitting nulls.
"""

import json
import types
from datetime import date
from decimal import Decimal
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from schema.fields import FIELD_GROUPS, LINE_ITEM_COLUMNS
from schema.po_schema import LineItem, PurchaseOrder

DATE_PATTERN = "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"


def _base_type(annotation: Any) -> Any:
    """Return T for `T | None`, otherwise the annotation unchanged."""
    if get_origin(annotation) in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) != 1:
            raise TypeError(f"Unsupported union type: {annotation}")
        return args[0]
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
        return {"type": "string", "pattern": DATE_PATTERN}
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
    cells = []
    for name in LINE_ITEM_COLUMNS:
        cell = _json_type(LineItem.model_fields[name].annotation)
        cell["type"] = [cell["type"], "null"]
        cells.append(cell)
    row = {
        "type": "array",
        "prefixItems": cells,
        "items": False,
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


def rows_to_line_items(rows: list[list[Any]]) -> list[LineItem]:
    """Convert compact rows (LINE_ITEM_COLUMNS order) to LineItem objects.

    Null cells are treated as absent. Raises MalformedRowError naming the row and the
    offending columns if a row has the wrong shape or fails LineItem validation.
    """
    items = []
    for index, row in enumerate(rows):
        if not isinstance(row, list):
            raise MalformedRowError(index, f"expected an array, got {type(row).__name__}")
        if len(row) != len(LINE_ITEM_COLUMNS):
            raise MalformedRowError(
                index, f"expected {len(LINE_ITEM_COLUMNS)} cells, got {len(row)}"
            )
        values = {
            name: cell
            for name, cell in zip(LINE_ITEM_COLUMNS, row, strict=True)
            if cell is not None
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


def parse_llm_json(text: str) -> Any:
    """Parse LLM JSON output, reading every non-integer number as Decimal (never float)."""
    return json.loads(text, parse_float=Decimal)
