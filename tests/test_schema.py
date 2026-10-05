"""Tests for the PO schema, field groups and LLM JSON schemas (Step 1.1)."""

import json
from datetime import date
from decimal import Decimal

import pytest

from schema.fields import (
    CRITICAL_FIELDS,
    CRITICAL_HEADER_FIELDS,
    CRITICAL_LINE_ITEM_FIELDS,
    FIELD_GROUPS,
    LINE_ITEM_COLUMNS,
)
from schema.llm_schemas import (
    DATE_PATTERN,
    MalformedRowError,
    compact_line_items_schema,
    json_schema_for_group,
    parse_llm_json,
    rows_to_line_items,
)
from schema.po_schema import LineItem, PurchaseOrder

HEADER_FIELDS = [name for name in PurchaseOrder.model_fields if name != "line_items"]

SAMPLE_ROW = [
    1,
    "BRG-6205",
    "Deep groove ball bearing 6205-2RS",
    "84821010",
    Decimal(120),
    "Nos",
    Decimal("245.50"),
    Decimal(5),
    Decimal("27987.00"),
    Decimal(18),
    Decimal("2518.83"),
    Decimal("2518.83"),
    None,
    Decimal("33024.66"),
    "2026-11-15",
]


# --- Model shape -------------------------------------------------------------------------


def test_field_counts_match_prd() -> None:
    assert len(HEADER_FIELDS) == 58
    assert len(LineItem.model_fields) == 15


def test_every_field_has_a_description() -> None:
    for model in (PurchaseOrder, LineItem):
        for name, field in model.model_fields.items():
            assert field.description, f"{model.__name__}.{name} has no description"


def test_critical_fields_are_exactly_the_required_fields() -> None:
    required_header = {
        name for name in HEADER_FIELDS if PurchaseOrder.model_fields[name].is_required()
    }
    required_line = {name for name, f in LineItem.model_fields.items() if f.is_required()}
    assert required_header == CRITICAL_HEADER_FIELDS
    assert required_line == CRITICAL_LINE_ITEM_FIELDS
    assert CRITICAL_FIELDS == CRITICAL_HEADER_FIELDS | CRITICAL_LINE_ITEM_FIELDS


def test_extra_fields_are_rejected() -> None:
    values = dict(zip(LINE_ITEM_COLUMNS, SAMPLE_ROW, strict=True))
    with pytest.raises(ValueError):
        LineItem.model_validate({**values, "made_up_field": "x"})


# --- Groups ------------------------------------------------------------------------------


def test_every_header_field_belongs_to_exactly_one_group() -> None:
    grouped = [name for fields in FIELD_GROUPS.values() for name in fields]
    assert sorted(grouped) == sorted(HEADER_FIELDS)
    assert len(grouped) == len(set(grouped)), "a field appears in more than one group"


def test_group_sizes() -> None:
    assert {gid: len(fields) for gid, fields in FIELD_GROUPS.items()} == {
        "G1_HEADER_TERMS": 20,
        "G2_PARTIES": 27,
        "G3_TOTALS": 11,
    }


def test_line_item_columns_follow_prd_order() -> None:
    assert LINE_ITEM_COLUMNS[0] == "line_no"
    assert LINE_ITEM_COLUMNS[-1] == "line_delivery_date"
    assert list(LineItem.model_fields) == LINE_ITEM_COLUMNS


# --- Group JSON schemas ------------------------------------------------------------------


@pytest.mark.parametrize("group_id", list(FIELD_GROUPS))
def test_group_schema_covers_only_its_fields(group_id: str) -> None:
    schema = json_schema_for_group(group_id)
    assert schema["type"] == "object"
    assert list(schema["properties"]) == FIELD_GROUPS[group_id]
    assert schema["additionalProperties"] is False
    assert "required" not in schema  # all optional: the model omits absent fields


@pytest.mark.parametrize("group_id", list(FIELD_GROUPS))
def test_group_schema_is_flat_and_portable(group_id: str) -> None:
    text = json.dumps(json_schema_for_group(group_id))  # also proves it is JSON-serialisable
    for keyword in ("$ref", "$defs", "anyOf", "oneOf", "allOf"):
        assert keyword not in text
    for prop in json_schema_for_group(group_id)["properties"].values():
        assert prop["type"] in ("string", "number", "integer")
        assert prop["description"]


def test_group_schema_types() -> None:
    totals = json_schema_for_group("G3_TOTALS")["properties"]
    assert totals["grand_total"]["type"] == "number"
    assert totals["amount_in_words"]["type"] == "string"
    header = json_schema_for_group("G1_HEADER_TERMS")["properties"]
    assert header["po_date"] == {**header["po_date"], "type": "string", "pattern": DATE_PATTERN}
    assert header["po_number"]["type"] == "string"
    parties = json_schema_for_group("G2_PARTIES")["properties"]
    assert parties["buyer_state_code"]["type"] == "string"  # keeps leading zeros, e.g. '07'


def test_descriptions_carry_label_synonyms() -> None:
    parties = json_schema_for_group("G2_PARTIES")["properties"]
    assert "Consignee" in parties["ship_to_name"]["description"]
    assert "Deliver To" in parties["ship_to_name"]["description"]
    assert "Supplier" in parties["vendor_name"]["description"]
    header = json_schema_for_group("G1_HEADER_TERMS")["properties"]
    assert "Order No." in header["po_number"]["description"]
    assert "P.O. Ref" in header["po_number"]["description"]


def test_unknown_group_raises() -> None:
    with pytest.raises(ValueError, match="Unknown group"):
        json_schema_for_group("G9_NOPE")


# --- Compact line items ------------------------------------------------------------------


def test_compact_schema_shape() -> None:
    schema = compact_line_items_schema()
    assert schema["required"] == ["rows"]
    row = schema["properties"]["rows"]["items"]
    assert len(row["prefixItems"]) == len(LINE_ITEM_COLUMNS) == 15
    assert row["minItems"] == row["maxItems"] == 15
    assert row["prefixItems"][0]["type"] == ["integer", "null"]
    assert row["prefixItems"][LINE_ITEM_COLUMNS.index("unit_rate")]["type"] == ["number", "null"]
    assert "$ref" not in json.dumps(schema)


def test_compact_row_round_trip() -> None:
    [item] = rows_to_line_items([SAMPLE_ROW])
    assert item.line_no == 1
    assert item.hsn_sac == "84821010"
    assert item.unit_rate == Decimal("245.50")
    assert item.igst_amount is None
    assert item.line_delivery_date == date(2026, 11, 15)
    back = [getattr(item, name) for name in LINE_ITEM_COLUMNS]
    assert back[:-1] == SAMPLE_ROW[:-1]
    assert back[-1] == date(2026, 11, 15)


def test_round_trip_from_llm_json_text() -> None:
    text = json.dumps({"rows": [SAMPLE_ROW]}, default=str).replace('"245.50"', "245.50")
    rows = parse_llm_json(text)["rows"]
    [item] = rows_to_line_items(rows)
    assert item.unit_rate == Decimal("245.50")


def test_wrong_row_length_raises_clear_error() -> None:
    with pytest.raises(MalformedRowError, match="row 1: expected 15 cells, got 14") as exc:
        rows_to_line_items([SAMPLE_ROW, SAMPLE_ROW[:-1]])
    assert exc.value.row_index == 1


def test_non_array_row_raises() -> None:
    with pytest.raises(MalformedRowError, match="row 0: expected an array, got dict"):
        rows_to_line_items([{"line_no": 1}])  # type: ignore[list-item]


def test_missing_critical_cell_names_the_column() -> None:
    row = list(SAMPLE_ROW)
    row[LINE_ITEM_COLUMNS.index("hsn_sac")] = None
    with pytest.raises(MalformedRowError, match="row 0: hsn_sac: Field required"):
        rows_to_line_items([row])


def test_bad_cell_value_names_the_column() -> None:
    row = list(SAMPLE_ROW)
    row[LINE_ITEM_COLUMNS.index("quantity")] = "twelve"
    with pytest.raises(MalformedRowError, match="row 0: quantity"):
        rows_to_line_items([row])


# --- Decimal precision -------------------------------------------------------------------


def test_parse_llm_json_never_produces_floats() -> None:
    value = parse_llm_json('{"grand_total": 1234567890123456.78, "round_off": -0.10}')
    assert value["grand_total"] == Decimal("1234567890123456.78")
    assert value["round_off"] == Decimal("-0.10")
    assert not any(isinstance(v, float) for v in value.values())


def test_decimal_precision_survives_the_model() -> None:
    text = (
        '{"rows": [[1, null, "Item", "8482", 0.1, null, 1234567890123456.78, null, 0.3, 18, '
        "null, null, null, 0.35, null]]}"
    )
    [item] = rows_to_line_items(parse_llm_json(text)["rows"])
    assert isinstance(item.unit_rate, Decimal)
    assert item.unit_rate == Decimal("1234567890123456.78")
    assert item.quantity + item.taxable_value == Decimal("0.4")  # exact, unlike 0.1 + 0.3 floats
    # The same text through plain json.loads loses digits; this is the bug parse_llm_json prevents.
    assert Decimal(str(json.loads(text)["rows"][0][6])) != Decimal("1234567890123456.78")


def test_decimal_preserved_in_json_dump() -> None:
    [item] = rows_to_line_items([SAMPLE_ROW])
    assert item.model_dump(mode="json")["unit_rate"] == "245.50"
