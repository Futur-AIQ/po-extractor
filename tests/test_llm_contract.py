"""Tests for the v1.1 LLM output contract (Step 1.4a, PRD §6.3).

Covers computed fields, short keys, the all-header schema, the reduced line-item schema and
reduced-row conversion. Step 1.1 behaviour is covered (unchanged) by test_schema.py.
"""

import json
import re
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from datagen.generate import Knobs, generate_po
from datagen.templates import format_date
from schema.fields import (
    COMPUTED_FIELDS,
    COMPUTED_HEADER_FIELDS,
    COMPUTED_LINE_ITEM_FIELDS,
    CRITICAL_FIELDS,
    HEADER_FIELDS,
    HEADER_FIELDS_FOR_LLM,
    LINE_ITEM_COLUMNS,
    LLM_LINE_ITEM_COLUMNS,
)
from schema.llm_keys import (
    FULL_KEYS,
    SHORT_KEY_MAX_LEN,
    SHORT_KEYS,
    to_full_keys,
    to_short_keys,
)
from schema.llm_schemas import (
    MalformedRowError,
    compact_line_items_schema,
    header_schema_for_llm,
    json_schema_for_group,
    line_item_columns_instruction,
    line_items_schema_for_llm,
    parse_llm_json,
    rows_to_line_items,
)
from schema.po_schema import LineItem, PurchaseOrder

# JSON-schema keywords both llama.cpp (json-schema-to-grammar) and vLLM (xgrammar) accept;
# all v1.1 schemas were also sent to llama.cpp during development and accepted.
SIMPLE_KEYWORDS = {
    "type", "properties", "additionalProperties", "required", "description", "items",
    "prefixItems", "minItems", "maxItems", "minLength",
}  # fmt: skip

FULL_PO = generate_po(
    21, Knobs(optional_drop_rate=0, p_amendment=1, p_quotation_ref=1, p_indent_no=1,
              p_freight=1, p_other_charges=1, p_header_discount=1, p_line_delivery_dates=1,
              p_discount_per_line=0.5),
).po  # fmt: skip


# --- Field lists -----------------------------------------------------------------------------


def test_computed_fields() -> None:
    assert {"total_tax"} == COMPUTED_HEADER_FIELDS
    assert {"cgst_amount", "sgst_amount", "igst_amount"} == COMPUTED_LINE_ITEM_FIELDS
    assert COMPUTED_FIELDS == COMPUTED_HEADER_FIELDS | COMPUTED_LINE_ITEM_FIELDS
    assert not COMPUTED_FIELDS & CRITICAL_FIELDS  # computed fields are never critical inputs
    assert set(HEADER_FIELDS) >= COMPUTED_HEADER_FIELDS
    assert set(LINE_ITEM_COLUMNS) >= COMPUTED_LINE_ITEM_FIELDS


def test_header_fields_for_llm_are_all_headers_minus_computed() -> None:
    assert [name for name in PurchaseOrder.model_fields if name != "line_items"] == HEADER_FIELDS
    assert len(HEADER_FIELDS) == 58
    assert [n for n in HEADER_FIELDS if n != "total_tax"] == HEADER_FIELDS_FOR_LLM
    assert len(HEADER_FIELDS_FOR_LLM) == 57


def test_llm_line_item_columns_keep_relative_order() -> None:
    assert LLM_LINE_ITEM_COLUMNS == [
        "line_no", "item_code", "description", "hsn_sac", "quantity", "uom", "unit_rate",
        "discount_pct", "taxable_value", "gst_rate", "line_total", "line_delivery_date",
    ]  # fmt: skip
    positions = [LINE_ITEM_COLUMNS.index(name) for name in LLM_LINE_ITEM_COLUMNS]
    assert positions == sorted(positions)


# --- Short keys ------------------------------------------------------------------------------


def test_short_keys_are_complete_unique_and_short() -> None:
    assert set(SHORT_KEYS) == set(PurchaseOrder.model_fields)
    shorts = list(SHORT_KEYS.values())
    assert len(shorts) == len(set(shorts)), "duplicate short key"
    for full, short in SHORT_KEYS.items():
        assert len(short) <= SHORT_KEY_MAX_LEN, (full, short)
        assert re.fullmatch(r"[a-z][a-z_]*", short), (full, short)
    assert {short: full for full, short in SHORT_KEYS.items()} == FULL_KEYS


@pytest.mark.parametrize(
    ("full", "short"),
    [("po_number", "po_no"), ("po_date", "po_dt"), ("vendor_gstin", "v_gstin"),
     ("buyer_gstin", "b_gstin"), ("ship_to_gstin", "st_gstin"), ("grand_total", "gr_total")],
)  # fmt: skip
def test_examples_from_the_contract(full: str, short: str) -> None:
    assert SHORT_KEYS[full] == short


def test_short_key_round_trip_on_a_full_purchase_order() -> None:
    data = FULL_PO.model_dump(mode="json")
    assert all(value is not None for value in data.values())  # every field exercised
    short = to_short_keys(data)
    assert set(short) == set(SHORT_KEYS.values())
    assert to_full_keys(short) == data
    # Through JSON text, as the LLM output would arrive, back to an identical model.
    text = json.dumps(short)
    assert PurchaseOrder.model_validate(to_full_keys(parse_llm_json(text))) == FULL_PO


def test_unknown_keys_raise() -> None:
    with pytest.raises(KeyError, match="unknown short key 'bogus'"):
        to_full_keys({"po_no": "1", "bogus": 2})
    with pytest.raises(KeyError, match="unknown field name 'bogus'"):
        to_short_keys({"bogus": 1})


# --- Schemas -----------------------------------------------------------------------------------


def _keywords(node: Any) -> set[str]:
    """Every JSON-schema keyword used anywhere in `node` (property names excluded)."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            if key == "properties":
                for sub in value.values():
                    found |= _keywords(sub)
            elif key in ("items", "prefixItems"):
                found |= _keywords(value)
    elif isinstance(node, list):
        for sub in node:
            found |= _keywords(sub)
    return found


def test_header_schema_has_all_non_computed_fields_with_short_keys() -> None:
    schema = header_schema_for_llm()
    assert list(schema["properties"]) == [SHORT_KEYS[n] for n in HEADER_FIELDS_FOR_LLM]
    assert schema["additionalProperties"] is False
    assert "required" not in schema  # all optional: absent fields are omitted
    for prop in schema["properties"].values():
        assert prop["description"]
        if prop["type"] == "string":
            assert prop["minLength"] == 1  # "" is not a way to say "absent"


def test_header_schema_descriptions_keep_label_synonyms() -> None:
    props = header_schema_for_llm()["properties"]
    assert "Consignee" in props["st_name"]["description"]
    assert "Supplier" in props["v_name"]["description"]
    assert "P.O. Ref" in props["po_no"]["description"]
    assert props["gr_total"]["type"] == "number"
    assert props["po_dt"]["type"] == "string" and "pattern" not in props["po_dt"]


def test_line_items_schema_uses_reduced_columns() -> None:
    row = line_items_schema_for_llm()["properties"]["rows"]["items"]
    assert len(row["prefixItems"]) == row["minItems"] == row["maxItems"] == 12
    types = [cell["type"] for cell in row["prefixItems"]]
    assert types[LLM_LINE_ITEM_COLUMNS.index("line_total")] == ["number", "null"]
    assert types[LLM_LINE_ITEM_COLUMNS.index("item_code")] == ["string", "null"]


def test_computed_fields_absent_from_both_llm_schemas() -> None:
    header = header_schema_for_llm()
    assert not {SHORT_KEYS[n] for n in COMPUTED_HEADER_FIELDS} & set(header["properties"])
    assert not COMPUTED_FIELDS & set(header["properties"])
    instruction = line_item_columns_instruction()
    for name in COMPUTED_LINE_ITEM_FIELDS:
        assert name not in LLM_LINE_ITEM_COLUMNS
        assert name not in instruction
    assert len(line_items_schema_for_llm()["properties"]["rows"]["items"]["prefixItems"]) == len(
        LINE_ITEM_COLUMNS
    ) - len(COMPUTED_LINE_ITEM_FIELDS)


@pytest.mark.parametrize(
    "schema",
    [header_schema_for_llm(), line_items_schema_for_llm(), compact_line_items_schema(),
     json_schema_for_group("G2_PARTIES")],
)  # fmt: skip
def test_schemas_use_only_simple_keywords(schema: dict) -> None:
    assert _keywords(schema) <= SIMPLE_KEYWORDS
    text = json.dumps(schema)
    for keyword in ("$ref", "$defs", "anyOf", "oneOf", "allOf", "pattern", "format"):
        assert f'"{keyword}"' not in text


def test_column_instruction_lists_columns_in_order() -> None:
    lines = line_item_columns_instruction().splitlines()
    assert "12 values" in lines[0] and "null" in lines[0]
    assert len(lines) == 1 + len(LLM_LINE_ITEM_COLUMNS)
    for number, (line, name) in enumerate(zip(lines[1:], LLM_LINE_ITEM_COLUMNS, strict=True), 1):
        assert line.startswith(f"{number}. {name}: ")
    assert "'Basic Value'" in lines[1 + LLM_LINE_ITEM_COLUMNS.index("taxable_value")]


# --- Reduced rows ------------------------------------------------------------------------------


def _as_llm_row(item: LineItem) -> list[Any]:
    """The row a model would emit: LLM columns, numbers as Decimal, dates as printed."""
    row = []
    for name in LLM_LINE_ITEM_COLUMNS:
        value = getattr(item, name)
        row.append(format_date(value, "dd/mm/yyyy") if isinstance(value, date) else value)
    return row


def test_reduced_row_round_trip() -> None:
    rows = [_as_llm_row(item) for item in FULL_PO.line_items]
    assert all(len(row) == 12 for row in rows)
    items = rows_to_line_items(rows)  # default: width 12 -> LLM columns
    assert items == rows_to_line_items(rows, columns=LLM_LINE_ITEM_COLUMNS)
    for original, item in zip(FULL_PO.line_items, items, strict=True):
        # Computed tax amounts are left for code to fill in; everything else is identical.
        assert item.cgst_amount is item.sgst_amount is item.igst_amount is None
        assert item == original.model_copy(update=dict.fromkeys(COMPUTED_LINE_ITEM_FIELDS))


def test_reduced_row_through_json_keeps_decimals() -> None:
    row = _as_llm_row(FULL_PO.line_items[0])
    text = json.dumps({"rows": [row]}, default=str)
    for value in (FULL_PO.line_items[0].unit_rate, FULL_PO.line_items[0].line_total):
        text = text.replace(f'"{value}"', str(value))  # numbers as JSON numbers, like the LLM
    [item] = rows_to_line_items(parse_llm_json(text)["rows"])
    assert item.unit_rate == FULL_PO.line_items[0].unit_rate
    assert isinstance(item.line_total, Decimal)


def test_full_width_rows_still_work_by_default() -> None:
    full_row = [getattr(FULL_PO.line_items[0], name) for name in LINE_ITEM_COLUMNS]
    [item] = rows_to_line_items([full_row])
    assert item == FULL_PO.line_items[0]


def test_explicit_columns_and_width_errors() -> None:
    row = _as_llm_row(FULL_PO.line_items[0])
    with pytest.raises(MalformedRowError, match="row 0: expected 15 cells, got 12"):
        rows_to_line_items([row], columns=LINE_ITEM_COLUMNS)
    with pytest.raises(MalformedRowError, match=r"expected 15 cells, got 11 \(12 also accepted"):
        rows_to_line_items([row[:-1]])
    bad = list(row)
    bad[LLM_LINE_ITEM_COLUMNS.index("taxable_value")] = None
    with pytest.raises(MalformedRowError, match="row 0: taxable_value: Field required"):
        rows_to_line_items([bad])
