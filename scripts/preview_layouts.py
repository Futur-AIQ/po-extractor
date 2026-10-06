"""Render one generated PO in all six layouts for visual review.

Run from the repo root:
    uv run python -m scripts.preview_layouts [--seed 2026]

Writes data/preview/<layout>.html plus data/preview/index.html linking them all. Open the
index in Chrome; use Print preview (Cmd+P) to see A4 pages, the repeated header and
"Page X of Y" footers.
"""

import argparse
import html
from pathlib import Path

from app.core.settings import get_settings
from datagen.generate import Knobs, generate_po
from datagen.templates import LAYOUTS, render_html

# A feature-rich PO, so every block of every layout is visible in the preview.
PREVIEW_KNOBS = Knobs(
    min_lines=32,
    max_lines=40,
    optional_drop_rate=0,
    p_amendment=1,
    p_quotation_ref=1,
    p_indent_no=1,
    p_freight=1,
    p_other_charges=1,
    p_header_discount=1,
    p_line_delivery_dates=1,
    p_discount_per_line=0.3,
    p_multiline_description_per_line=0.4,
)

DESCRIPTIONS = {
    "L1_classic_erp": "Boxed header grid, buyer/vendor side by side, full per-line tax columns",
    "L2_tally_style": "Dense bordered Tally table, CGST/SGST/IGST columns, HSN-wise tax summary",
    "L3_modern_minimal": "No borders, key-value header, vendor block on the right, light colours",
    "L4_engineering": "Material codes, drawing/spec refs, multi-line descriptions, delivery dates",
    "L5_label_variants": "Synonym labels: Order Ref, Supplier, Consignee, Invoice To, Rate/Unit",
    "L6_psu_formal": "Formal letter opening, items, totals, numbered terms & conditions page",
}


def main() -> None:
    """Render the preview PO in every layout and write an index page."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = Path(get_settings().data_dir) / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_po(args.seed, PREVIEW_KNOBS)
    po = generated.po

    links = []
    for layout_id in LAYOUTS:
        path = out_dir / f"{layout_id}.html"
        path.write_text(render_html(po, layout_id, tc_page=True), encoding="utf-8")
        links.append(
            f'<li><a href="{layout_id}.html">{layout_id}</a> - '
            f"{html.escape(DESCRIPTIONS[layout_id])}</li>"
        )
        print(f"wrote {path}")

    regime = "inter-state (IGST)" if generated.meta.inter_state else "intra-state (CGST+SGST)"
    index = out_dir / "index.html"
    index.write_text(
        "<!DOCTYPE html><meta charset='utf-8'><title>PO layout preview</title>"
        "<body style='font:15px system-ui;margin:40px;max-width:900px'>"
        f"<h1>PO layout preview</h1><p>PO <b>{html.escape(po.po_number)}</b>, seed {args.seed}, "
        f"{len(po.line_items)} lines, {regime}, grand total Rs {po.grand_total}. "
        "All six files render the same ground truth. Use Print preview (Cmd+P) to see pages.</p>"
        f"<ul>{''.join(links)}</ul></body>",
        encoding="utf-8",
    )
    print(f"wrote {index}")


if __name__ == "__main__":
    main()
