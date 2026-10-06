"""Build the synthetic PO dataset: PDFs, ground truth, manifest, splits and a visual preview.

    uv run python -m datagen.build --n 60 --seed 42 --out data/synthetic

Output layout:
    pdfs/PO_0001.pdf      rendered purchase order
    truth/PO_0001.json    {"po": PurchaseOrder, "meta": {layout, seed, knobs, page_count, ...}}
    thumbs/PO_0001.png    page 1 thumbnail (for preview.html)
    manifest.csv          one row per PO
    splits.json           {"dev": [...], "test": [...]}, stratified by layout
    preview.html          thumbnail grid for quick visual QA

Page-count control: a PO must render to 2-3 pages (2-4 if it has a T&C page). If not, the
line count is adjusted and the PO regenerated from the same seed (deterministic). After
MAX_ATTEMPTS the seed is logged and skipped, and the slot moves to its next seed.
"""

import argparse
import asyncio
import csv
import dataclasses
import html
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from app.core.logging import configure_logging, get_logger
from app.core.settings import get_settings
from datagen.generate import GeneratedPO, Knobs, generate_po
from datagen.render import PdfRenderer, lines_per_page, page_texts
from datagen.templates import LAYOUTS, LAYOUTS_WITH_FIXED_TC_PAGE, render_generated
from schema.po_schema import PurchaseOrder

log = get_logger(__name__)

LAYOUT_IDS = list(LAYOUTS)
MIN_PAGES, MAX_PAGES = 2, 3  # +1 allowed when the PO has a T&C page
MAX_ATTEMPTS = 5  # line-count adjustments per seed
MAX_SEEDS_PER_SLOT = 5  # seeds tried before the build fails
LINE_COUNT_FLOOR, LINE_COUNT_CEILING = 8, 90
GENERATED_OUTPUTS = ("pdfs", "truth", "thumbs", "manifest.csv", "splits.json", "preview.html")


@dataclass(frozen=True)
class BuiltPO:
    """One rendered PO that passed page-count control."""

    po_id: str
    layout: str
    generated: GeneratedPO
    knobs: Knobs
    pdf: bytes
    page_count: int
    lines_per_page: list[list[int]]
    attempts: int
    has_tc_page: bool

    @property
    def po(self) -> PurchaseOrder:
        return self.generated.po


# =========================================================================================
# Page-count control
# =========================================================================================


def page_range(has_tc_page: bool) -> tuple[int, int]:
    """Allowed page counts: 2-3, or 2-4 when a T&C page is appended."""
    return MIN_PAGES, MAX_PAGES + (1 if has_tc_page else 0)


def next_line_count(lines: int, pages: int, low: int, high: int) -> int:
    """Line count to try next, scaled towards the middle of the allowed page range."""
    proposed = round(lines * ((low + high) / 2) / pages)
    proposed = min(proposed, lines - 1) if pages > high else max(proposed, lines + 1)
    return max(LINE_COUNT_FLOOR, min(LINE_COUNT_CEILING, proposed))


async def build_one(
    renderer: PdfRenderer, po_id: str, layout: str, seed: int, knobs: Knobs
) -> BuiltPO | None:
    """Render one seed in `layout`, adjusting the line count until the page count fits."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        generated = generate_po(seed, knobs)
        has_tc_page = generated.meta.has_tc_page or layout in LAYOUTS_WITH_FIXED_TC_PAGE
        html_doc = await asyncio.to_thread(render_generated, generated, layout)
        pdf = await renderer.render(html_doc)
        texts = await asyncio.to_thread(page_texts, pdf)
        low, high = page_range(has_tc_page)
        lines = len(generated.po.line_items)
        if low <= len(texts) <= high:
            return BuiltPO(
                po_id, layout, generated, knobs, pdf, len(texts),
                lines_per_page(generated.po, texts), attempt, has_tc_page,
            )  # fmt: skip
        if lines in (LINE_COUNT_FLOOR, LINE_COUNT_CEILING):
            break  # cannot move further in the needed direction
        new_lines = next_line_count(lines, len(texts), low, high)
        log.info(
            "page count out of range, adjusting lines",
            extra={"po_id": po_id, "layout": layout, "seed": seed, "pages": len(texts),
                   "allowed": [low, high], "lines": lines, "next_lines": new_lines},
        )  # fmt: skip
        knobs = dataclasses.replace(knobs, min_lines=new_lines, max_lines=new_lines)
    log.warning(
        "skipping seed: page count not reachable",
        extra={"po_id": po_id, "layout": layout, "seed": seed},
    )
    return None


async def build_slot(renderer: PdfRenderer, index: int, seed: int, knobs: Knobs) -> BuiltPO:
    """Build dataset slot `index` (layout round-robin); seeds derive from (seed, index)."""
    po_id = f"PO_{index + 1:04d}"
    layout = LAYOUT_IDS[index % len(LAYOUT_IDS)]
    seeds = random.Random(f"{seed}:{index}")  # str seeds are stable across runs
    for _ in range(MAX_SEEDS_PER_SLOT):
        built = await build_one(renderer, po_id, layout, seeds.getrandbits(63), knobs)
        if built is not None:
            return built
    raise RuntimeError(f"{po_id}: no seed rendered to an allowed page count")


# =========================================================================================
# Splits
# =========================================================================================


def make_splits(layout_by_id: dict[str, str], dev_size: int, seed: int) -> dict[str, list[str]]:
    """Dev/test split stratified by layout: dev takes POs round-robin across layouts."""
    rng = random.Random(f"splits:{seed}")
    by_layout: dict[str, list[str]] = {}
    for po_id in sorted(layout_by_id):
        by_layout.setdefault(layout_by_id[po_id], []).append(po_id)
    for ids in by_layout.values():
        rng.shuffle(ids)

    dev: list[str] = []
    while len(dev) < min(dev_size, len(layout_by_id)):
        for layout in sorted(by_layout):
            if by_layout[layout] and len(dev) < dev_size:
                dev.append(by_layout[layout].pop())
    test = sorted(set(layout_by_id) - set(dev))
    return {"dev": sorted(dev), "test": test}


# =========================================================================================
# Writing outputs
# =========================================================================================


def truth_document(built: BuiltPO) -> dict[str, Any]:
    """The ground-truth JSON for one PO."""
    meta = built.generated.meta
    return {
        "po": built.po.model_dump(mode="json"),
        "meta": {
            "id": built.po_id,
            "layout": built.layout,
            "seed": meta.seed,
            "knobs": dataclasses.asdict(built.knobs),
            "page_count": built.page_count,
            "line_count": len(built.po.line_items),
            "lines_per_page": built.lines_per_page,
            "attempts": built.attempts,
            "has_tc_page": built.has_tc_page,
            "rows_may_break": meta.rows_may_break,
            "inter_state": meta.inter_state,
            "vendor_industry": meta.vendor_industry,
            "issuer_id": meta.issuer_id,
            "issuer_frequent": meta.issuer_frequent,
            "expected_duplicate": meta.expected_duplicate,
        },
    }


def read_truth(path: Path) -> tuple[PurchaseOrder, dict[str, Any]]:
    """Load a truth file back into a PurchaseOrder (Decimals exact) and its meta dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return PurchaseOrder.model_validate(data["po"]), data["meta"]


def write_po(out: Path, built: BuiltPO) -> None:
    """Write the PDF, truth JSON and page-1 thumbnail of one PO."""
    (out / "pdfs" / f"{built.po_id}.pdf").write_bytes(built.pdf)
    (out / "truth" / f"{built.po_id}.json").write_text(
        json.dumps(truth_document(built), indent=2, default=str) + "\n", encoding="utf-8"
    )
    with pymupdf.open(stream=built.pdf, filetype="pdf") as doc:
        doc[0].get_pixmap(dpi=45).save(out / "thumbs" / f"{built.po_id}.png")


def write_manifest(out: Path, built: list[BuiltPO], split_of: dict[str, str]) -> None:
    fields = [
        "id", "layout", "split", "seed", "page_count", "line_count", "has_tc_page",
        "rows_may_break", "inter_state", "po_number", "grand_total", "pdf", "truth",
    ]  # fmt: skip
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for b in built:
            writer.writerow(
                {
                    "id": b.po_id,
                    "layout": b.layout,
                    "split": split_of[b.po_id],
                    "seed": b.generated.meta.seed,
                    "page_count": b.page_count,
                    "line_count": len(b.po.line_items),
                    "has_tc_page": b.has_tc_page,
                    "rows_may_break": b.generated.meta.rows_may_break,
                    "inter_state": b.generated.meta.inter_state,
                    "po_number": b.po.po_number,
                    "grand_total": str(b.po.grand_total),
                    "pdf": f"pdfs/{b.po_id}.pdf",
                    "truth": f"truth/{b.po_id}.json",
                }
            )


def write_preview(out: Path, built: list[BuiltPO], split_of: dict[str, str]) -> None:
    cards = "".join(
        f'<figure><a href="pdfs/{b.po_id}.pdf"><img src="thumbs/{b.po_id}.png" loading="lazy"></a>'
        f"<figcaption><b>{b.po_id}</b> {html.escape(b.layout)}<br>{b.page_count} pages, "
        f"{len(b.po.line_items)} lines, {split_of[b.po_id]}"
        f"{', IGST' if b.generated.meta.inter_state else ''}"
        f"{', T&amp;C' if b.has_tc_page else ''}</figcaption></figure>"
        for b in built
    )
    (out / "preview.html").write_text(
        "<!DOCTYPE html><meta charset='utf-8'><title>Synthetic PO dataset</title><style>"
        "body{font:13px system-ui;margin:20px;background:#eef0f3}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px}"
        "figure{margin:0;background:#fff;padding:6px;box-shadow:0 1px 4px #0003}"
        "img{width:100%;display:block;border:1px solid #ddd}figcaption{margin-top:4px}</style>"
        f"<h1>Synthetic PO dataset ({len(built)} POs)</h1><main>{cards}</main>",
        encoding="utf-8",
    )


def prepare_output_dir(out: Path) -> None:
    """Create `out`, removing only outputs a previous build wrote there."""
    out.mkdir(parents=True, exist_ok=True)
    for name in GENERATED_OUTPUTS:
        target = out / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for sub in ("pdfs", "truth", "thumbs"):
        (out / sub).mkdir()


# =========================================================================================
# Entry points
# =========================================================================================


async def build_dataset(
    n: int,
    seed: int,
    out: Path,
    *,
    dev_size: int = 10,
    knobs: Knobs | None = None,
    concurrency: int = 4,
) -> list[BuiltPO]:
    """Build `n` POs into `out` and write all dataset files. Returns the built POs."""
    knobs = knobs or Knobs()
    prepare_output_dir(out)
    async with PdfRenderer(max_pages=concurrency) as renderer:
        built = await asyncio.gather(*(build_slot(renderer, i, seed, knobs) for i in range(n)))

    for b in built:
        await asyncio.to_thread(write_po, out, b)
        unmapped = len(b.po.line_items) - sum(len(page) for page in b.lines_per_page)
        if unmapped:
            log.warning("lines not found on any page", extra={"po_id": b.po_id, "count": unmapped})

    splits = make_splits({b.po_id: b.layout for b in built}, dev_size, seed)
    split_of = {po_id: name for name, ids in splits.items() for po_id in ids}
    (out / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    write_manifest(out, built, split_of)
    write_preview(out, built, split_of)
    return built


def main() -> None:
    """CLI entry point."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the synthetic PO dataset.")
    parser.add_argument("--n", type=int, default=60, help="number of POs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(settings.data_dir) / "synthetic")
    parser.add_argument("--dev", type=int, default=10, help="size of the dev split")
    parser.add_argument("--concurrency", type=int, default=4, help="concurrent browser tabs")
    args = parser.parse_args()
    configure_logging(settings.log_level)

    start = time.perf_counter()
    built = asyncio.run(
        build_dataset(args.n, args.seed, args.out, dev_size=args.dev, concurrency=args.concurrency)
    )
    pages = [b.page_count for b in built]
    line_counts = [len(b.po.line_items) for b in built]
    log.info(
        "dataset built",
        extra={
            "out": str(args.out),
            "pos": len(built),
            "seconds": round(time.perf_counter() - start, 1),
            "pages": {str(p): pages.count(p) for p in sorted(set(pages))},
            "lines_min_max": [min(line_counts), max(line_counts)],
            "regenerated": sum(b.attempts > 1 for b in built),
        },
    )
    print(f"Open {args.out / 'preview.html'} for a visual check.")


if __name__ == "__main__":
    main()
