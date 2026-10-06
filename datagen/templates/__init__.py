"""Render a PurchaseOrder as A4 print-ready HTML in one of six layouts.

    render_html(po, "L2_tally_style", tc_page=True, rows_may_break=False) -> str

Formatting rules live here, not in the templates:
- amounts use Indian digit grouping (1,23,456.78); quantities keep their own precision
- each layout prints dates in its own style; the ground truth stays ISO
- fields that are None are never rendered (the macros skip them, so no empty labels)
- every field that is present in the PO is printed somewhere, so the ground truth is fair

Templates: base.html (page CSS, repeated header, page numbers), _macros.html, _tc_page.html,
and one file per layout.
"""

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markupsafe import Markup

from datagen.generate import GeneratedPO
from schema.po_schema import LineItem, PurchaseOrder

TEMPLATE_DIR = Path(__file__).parent

# Layout id -> date style. Every style must be parseable by schema.dates.parse_po_date.
LAYOUTS: dict[str, str] = {
    "L1_classic_erp": "dd-mm-yyyy",
    "L2_tally_style": "dd-Mon-yyyy",
    "L3_modern_minimal": "dd Mon yyyy",
    "L4_engineering": "dd.mm.yyyy",
    "L5_label_variants": "dd/mm/yyyy",
    "L6_psu_formal": "dd/mm/yyyy",
}
LAYOUTS_WITH_FIXED_TC_PAGE = {"L6_psu_formal"}

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")  # fmt: skip


# =========================================================================================
# Formatting
# =========================================================================================


def format_indian(value: Decimal, places: int | None = 2) -> str:
    """Indian digit grouping: 1234567.8 -> '12,34,567.80'. places=None keeps the value's scale."""
    if places is not None:
        value = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    sign = "-" if value < 0 else ""
    whole, _, fraction = f"{abs(value):f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        pairs = []
        while len(head) > 2:
            pairs.insert(0, head[-2:])
            head = head[:-2]
        whole = ",".join([head, *pairs, tail])
    return f"{sign}{whole}.{fraction}" if fraction else f"{sign}{whole}"


def format_money(value: Decimal | None) -> str:
    """Amount with 2 decimals and Indian grouping; '' for None (so blank cells stay blank)."""
    return "" if value is None else format_indian(value, 2)


def format_quantity(value: Decimal | None) -> str:
    """Quantity exactly as stored (no forced decimals), with Indian grouping."""
    return "" if value is None else format_indian(value, None)


def format_percent(value: Decimal | None) -> str:
    """Percentage without trailing zeros: 18 -> '18', 2.50 -> '2.5'."""
    return "" if value is None else f"{value.normalize():f}"


def format_date(value: date | None, style: str) -> str:
    """Print a date in a layout's style. All styles parse back with schema.dates."""
    if value is None:
        return ""
    month = _MONTHS[value.month - 1]
    styles = {
        "dd-mm-yyyy": f"{value:%d-%m-%Y}",
        "dd/mm/yyyy": f"{value:%d/%m/%Y}",
        "dd.mm.yyyy": f"{value:%d.%m.%Y}",
        "dd-Mon-yyyy": f"{value.day:02d}-{month[:3]}-{value.year}",
        "dd Mon yyyy": f"{value.day} {month[:3]} {value.year}",
        "long": f"{_ordinal(value.day)} {month} {value.year}",
    }
    if style not in styles:
        raise ValueError(f"unknown date style {style!r}")
    return styles[style]


def _ordinal(day: int) -> str:
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def css_string(text: str) -> Markup:
    """Text for a CSS `content: "..."` value inside <style> (raw text, so no HTML escaping)."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("<", "\\3C ")
    return Markup(escaped)


def address_lines(address: str | None) -> list[str]:
    """Split a one-line address at ', ' into printed lines (joining them back gives the truth)."""
    return address.split(", ") if address else []


# =========================================================================================
# Views (plain data the templates lay out)
# =========================================================================================


@dataclass(frozen=True)
class PartyView:
    """One address block. Fields that are None are skipped by the party macro."""

    name: str | None
    address: list[str]
    gstin: str | None
    pan: str | None = None
    state: str | None = None
    state_code: str | None = None
    contact: str | None = None
    phone: str | None = None
    email: str | None = None
    code: str | None = None  # vendor code


def _parties(po: PurchaseOrder) -> dict[str, PartyView]:
    return {
        "buyer": PartyView(
            po.buyer_name, address_lines(po.buyer_address), po.buyer_gstin, po.buyer_pan,
            po.buyer_state, po.buyer_state_code, po.buyer_contact_person, po.buyer_phone,
            po.buyer_email,
        ),
        "vendor": PartyView(
            po.vendor_name, address_lines(po.vendor_address), po.vendor_gstin, po.vendor_pan,
            po.vendor_state, po.vendor_state_code, po.vendor_contact_person, po.vendor_phone,
            po.vendor_email, po.vendor_code,
        ),
        "bill_to": PartyView(
            po.bill_to_name, address_lines(po.bill_to_address), po.bill_to_gstin,
            state_code=po.bill_to_state_code,
        ),
        "ship_to": PartyView(
            po.ship_to_name, address_lines(po.ship_to_address), po.ship_to_gstin,
            state_code=po.ship_to_state_code,
        ),
    }  # fmt: skip


@dataclass(frozen=True)
class Columns:
    """Which optional item-table columns this PO needs (a column with no values is omitted)."""

    item_code: bool
    uom: bool
    discount: bool
    delivery_date: bool
    intra_state: bool  # CGST + SGST columns; otherwise IGST


def _columns(po: PurchaseOrder) -> Columns:
    items = po.line_items
    return Columns(
        item_code=any(i.item_code for i in items),
        uom=any(i.uom for i in items),
        discount=any(i.discount_pct is not None for i in items),
        delivery_date=any(i.line_delivery_date for i in items),
        intra_state=any(i.cgst_amount is not None for i in items),
    )


@dataclass(frozen=True)
class HsnRow:
    """One row of the HSN-wise tax summary (L2)."""

    hsn_sac: str
    gst_rate: Decimal
    taxable: Decimal
    cgst: Decimal
    sgst: Decimal
    igst: Decimal

    @property
    def total_tax(self) -> Decimal:
        return self.cgst + self.sgst + self.igst


def hsn_summary(items: list[LineItem]) -> list[HsnRow]:
    """Taxable value and taxes grouped by (HSN/SAC, GST rate), in first-seen order."""
    zero = Decimal("0.00")
    groups: dict[tuple[str, Decimal], list[LineItem]] = {}
    for item in items:
        groups.setdefault((item.hsn_sac, item.gst_rate), []).append(item)
    return [
        HsnRow(
            hsn,
            rate,
            sum((i.taxable_value for i in rows), zero),
            sum((i.cgst_amount or zero for i in rows), zero),
            sum((i.sgst_amount or zero for i in rows), zero),
            sum((i.igst_amount or zero for i in rows), zero),
        )
        for (hsn, rate), rows in groups.items()
    ]


def spec_reference(po: PurchaseOrder, item: LineItem) -> str:
    """Deterministic drawing/spec reference for L4 (a realistic column that is not a PO field)."""
    code = item.item_code or item.hsn_sac
    return f"SPEC-{item.hsn_sac[:4]}-{(sum(map(ord, code + po.po_number)) % 900) + 100}"


# =========================================================================================
# Rendering
# =========================================================================================

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html"]),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
_env.filters.update(
    money=format_money,
    qty=format_quantity,
    pct=format_percent,
    css_string=css_string,
)


def render_html(
    po: PurchaseOrder,
    layout_id: str,
    *,
    tc_page: bool = False,
    rows_may_break: bool = False,
) -> str:
    """Render `po` as a complete A4 HTML document in layout `layout_id`.

    tc_page: add a terms & conditions page (always on for L6).
    rows_may_break: allow item rows to split across pages (otherwise rows are kept whole).
    """
    if layout_id not in LAYOUTS:
        raise ValueError(f"unknown layout {layout_id!r}; expected one of {sorted(LAYOUTS)}")
    date_style = LAYOUTS[layout_id]
    return _env.get_template(f"{layout_id}.html").render(
        po=po,
        layout_id=layout_id,
        tc_page=tc_page or layout_id in LAYOUTS_WITH_FIXED_TC_PAGE,
        rows_may_break=rows_may_break,
        parties=_parties(po),
        cols=_columns(po),
        hsn_rows=hsn_summary(po.line_items),
        spec_ref=lambda item: spec_reference(po, item),
        d=lambda value: format_date(value, date_style),
        long_date=lambda value: format_date(value, "long"),
    )


def render_generated(generated: GeneratedPO, layout_id: str) -> str:
    """Render a generated PO using its own T&C and row-break knobs."""
    return render_html(
        generated.po,
        layout_id,
        tc_page=generated.meta.has_tc_page,
        rows_may_break=generated.meta.rows_may_break,
    )
