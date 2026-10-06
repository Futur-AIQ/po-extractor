"""Tests for the six HTML layouts (datagen/templates, Step 1.4)."""

import re
from datetime import date
from decimal import Decimal
from html import unescape

import pytest

from datagen.generate import GeneratedPO, Knobs, generate_many, generate_po
from datagen.templates import (
    LAYOUTS,
    css_string,
    format_date,
    format_indian,
    format_money,
    format_percent,
    format_quantity,
    hsn_summary,
    render_generated,
    render_html,
)
from schema.dates import parse_po_date
from schema.po_schema import LineItem, PurchaseOrder

RICH = Knobs(
    min_lines=8, max_lines=12, optional_drop_rate=0, p_amendment=1, p_quotation_ref=1,
    p_indent_no=1, p_freight=1, p_other_charges=1, p_header_discount=1,
    p_line_delivery_dates=1, p_discount_per_line=0.5, p_multiline_description_per_line=0.5,
)  # fmt: skip
BARE = Knobs(
    min_lines=5, max_lines=8, optional_drop_rate=1, p_amendment=0, p_quotation_ref=0,
    p_indent_no=0, p_freight=0, p_other_charges=0, p_header_discount=0,
    p_line_delivery_dates=0, p_discount_per_line=0,
)  # fmt: skip


def rich_pos() -> list[GeneratedPO]:
    """Feature-rich POs, intra- and inter-state, plus a few with default knobs."""
    return [
        generate_po(1, Knobs(**{**RICH.__dict__, "p_inter_state": 0})),
        generate_po(2, Knobs(**{**RICH.__dict__, "p_inter_state": 1})),
        *generate_many(4, seed=3, knobs=Knobs(min_lines=6, max_lines=10)),
    ]


def visible_text(html: str) -> str:
    """Text a reader sees in the body: tags removed, entities decoded, whitespace collapsed."""
    body = html.split("<body", 1)[1]
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", body)).split())


def item_rows(html: str) -> list[str]:
    """Visible text of each row in the items table body."""
    tbody = html.split('<table class="items">', 1)[1].split("<tbody>", 1)[1].split("</tbody>")[0]
    return [visible_text("<body>" + row) for row in tbody.split("<tr")[1:]]


def header_expectations(po: PurchaseOrder, layout: str) -> dict[str, str]:
    """For every present header field, the string that must appear in the document."""
    out = {}
    for name in PurchaseOrder.model_fields:
        value = getattr(po, name)
        if name == "line_items" or value is None:
            continue
        if isinstance(value, date):
            out[name] = format_date(value, LAYOUTS[layout])
        elif isinstance(value, Decimal):
            out[name] = format_money(value)
        elif name.endswith("_address"):
            out[name] = value.replace(", ", " ")  # printed one part per line
        else:
            out[name] = value
    return out


def line_expectations(item: LineItem, layout: str) -> dict[str, str]:
    """For every present line-item field, the string that must appear in its row."""
    out = {}
    for name in LineItem.model_fields:
        value = getattr(item, name)
        if value is None:
            continue
        if isinstance(value, date):
            out[name] = format_date(value, LAYOUTS[layout])
        elif name in ("quantity",):
            out[name] = format_quantity(value)
        elif name in ("discount_pct", "gst_rate"):
            out[name] = format_percent(value)
        elif isinstance(value, Decimal):
            out[name] = format_money(value)
        else:
            out[name] = str(value)
    return out


# --- Every present field is printed (the ground truth is fair) -------------------------------


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_every_present_field_is_rendered(layout: str) -> None:
    for g in rich_pos():
        html = render_generated(g, layout)
        text = visible_text(html)
        missing = {k: v for k, v in header_expectations(g.po, layout).items() if v not in text}
        assert missing == {}, f"{layout} seed {g.meta.seed}: header fields not printed"

        rows = item_rows(html)
        assert len(rows) == len(g.po.line_items)
        for item, row in zip(g.po.line_items, rows, strict=True):
            missing = {k: v for k, v in line_expectations(item, layout).items() if v not in row}
            assert missing == {}, f"{layout} line {item.line_no}: fields not in its row"


# --- Absent fields leave no trace ---------------------------------------------------------------

ABSENT_LABELS = {
    "L1_classic_erp": ["Amendment No.", "Quotation Ref.", "Indent No.", "Item Code", "Disc %",
                       "Delivery Date", "Freight Charges", "Less: Discount", "Vendor Code"],
    "L2_tally_style": ["Amendment No.", "Supplier's Ref.", "Indent No.", "Part No.", "Disc. %",
                       "Due on", "Freight Charges", "Ledger Code"],
    "L3_modern_minimal": ["Amendment", "Quote Ref", "Requisition", "Code:", "Disc", "Due",
                          "Other charges", "Vendor Code"],
    "L4_engineering": ["Amendment / Rev.", "Offer No.", "PR / Indent No.", "Material Code",
                       "Disc %", "Delivery Date", "Special Discount", "Supplier Code"],
    "L5_label_variants": ["Revision", "Your Ref", "Requisition No.", "Part No.", "Disc.",
                          "Schedule", "Less: Rebate", "Supplier ID"],
    "L6_psu_formal": ["Amendment No.", "Quotation dated", "Indent No.", "Code No.", "Disc. %",
                      "Delivery by", "Add: Freight", "Vendor Code"],
}  # fmt: skip


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_absent_fields_do_not_render(layout: str) -> None:
    for g in generate_many(4, seed=10, knobs=BARE):
        html = render_html(g.po, layout)
        text = visible_text(html)
        assert "None" not in html
        assert not re.search(r'class="v[^"]*">\s*</', html), "empty value next to a label"
        present = [
            label
            for label in ABSENT_LABELS[layout]
            if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text)
        ]
        assert present == [], f"{layout}: labels rendered for absent fields"


# --- Page furniture and knobs ---------------------------------------------------------------------


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_print_css(layout: str) -> None:
    html = render_html(generate_po(4).po, layout)
    assert "size: A4" in html
    assert 'content: "Page " counter(page) " of " counter(pages)' in html
    assert "@top-left" in html  # company header repeated on every page
    assert "table.items thead { display: table-header-group; }" in html
    assert "table.items tr { break-inside: avoid;" in html


@pytest.mark.parametrize("layout", [layout for layout in LAYOUTS if layout != "L6_psu_formal"])
def test_tc_page_follows_knob(layout: str) -> None:
    po = generate_po(5).po
    assert 'class="tc-page"' in render_html(po, layout, tc_page=True)
    assert 'class="tc-page"' not in render_html(po, layout, tc_page=False)


def test_psu_layout_always_has_tc_page_with_terms() -> None:
    po = generate_po(6, Knobs(optional_drop_rate=0)).po
    html = render_html(po, "L6_psu_formal", tc_page=False)
    assert 'class="tc-page"' in html
    tc = visible_text("<body>" + html.split('class="tc-page"', 1)[1])
    assert po.payment_terms in tc and po.warranty_terms in tc


def test_rows_may_break_knob() -> None:
    po = generate_po(7).po
    assert (
        "rows-may-break"
        in render_html(po, "L1_classic_erp", rows_may_break=True).split("<body", 1)[1][:80]
    )
    assert "rows-may-break" not in render_html(po, "L1_classic_erp").split("<body", 1)[1][:80]


def test_render_generated_uses_meta_knobs() -> None:
    g = generate_po(8, Knobs(p_tc_page=1, p_rows_split_across_pages=1))
    assert g.meta.has_tc_page and g.meta.rows_may_break
    html = render_generated(g, "L3_modern_minimal")
    assert 'class="tc-page"' in html and "rows-may-break" in html


# --- Layout diversity ---------------------------------------------------------------------------


def test_layouts_have_distinct_table_headers() -> None:
    po = generate_po(9, RICH).po
    headers = set()
    for layout in LAYOUTS:
        html = render_html(po, layout)
        thead = html.split('<table class="items">', 1)[1].split("</thead>")[0]
        headers.add(visible_text("<body>" + thead))
    assert len(headers) == len(LAYOUTS)


def test_label_variants_layout_uses_synonyms() -> None:
    text = visible_text(render_html(generate_po(10, RICH).po, "L5_label_variants"))
    for label in ("Order Ref", "Supplier", "Consignee", "Invoice To", "Rate/Unit", "Basic Amount"):
        assert label in text


def test_tally_layout_has_consistent_hsn_summary() -> None:
    po = generate_po(11, RICH).po
    rows = hsn_summary(po.line_items)
    assert sum(r.taxable for r in rows) == po.subtotal
    assert sum(r.total_tax for r in rows) == po.cgst_total + po.sgst_total + po.igst_total
    assert "HSN/SAC wise Tax Summary" in render_html(po, "L2_tally_style")


def test_psu_layout_is_a_letter() -> None:
    text = visible_text(render_html(generate_po(12).po, "L6_psu_formal"))
    assert "Dear Sir / Madam" in text and "Yours faithfully" in text


def test_rendering_is_deterministic() -> None:
    po = generate_po(13).po
    assert render_html(po, "L2_tally_style") == render_html(po, "L2_tally_style")


def test_unknown_layout_raises() -> None:
    with pytest.raises(ValueError, match="unknown layout"):
        render_html(generate_po(14).po, "L9_nope")


# --- Formatting ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "places", "text"),
    [
        ("123456.78", 2, "1,23,456.78"),
        ("123456789", 2, "12,34,56,789.00"),
        ("999.5", 2, "999.50"),
        ("1000", 2, "1,000.00"),
        ("-1234567.891", 2, "-12,34,567.89"),
        ("0.06", 2, "0.06"),
        ("1209.0", None, "1,209.0"),
        ("501", None, "501"),
    ],
)
def test_indian_number_format(value: str, places: int | None, text: str) -> None:
    assert format_indian(Decimal(value), places) == text


def test_percent_format() -> None:
    assert format_percent(Decimal("18")) == "18"
    assert format_percent(Decimal("2.50")) == "2.5"
    assert format_percent(Decimal("10")) == "10"


@pytest.mark.parametrize("style", sorted(set(LAYOUTS.values()) | {"long"}))
def test_every_date_style_parses_back(style: str) -> None:
    for day in (date(2026, 3, 1), date(2025, 11, 22), date(2026, 12, 31), date(2026, 2, 12)):
        assert parse_po_date(format_date(day, style)) == day


def test_css_string_escapes_quotes() -> None:
    assert css_string('A "B" \\ C') == 'A \\"B\\" \\\\ C'
