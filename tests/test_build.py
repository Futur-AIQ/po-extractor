"""Tests for PDF rendering and the dataset build (Step 1.5).

The build tests launch headless Chromium (local, no network). They are skipped if the
browser is not installed: run `uv run playwright install chromium`.
"""

import asyncio
import csv
import json
from pathlib import Path

import pymupdf
import pytest
from playwright.async_api import Error as PlaywrightError

from datagen.build import (
    LAYOUT_IDS,
    BuiltPO,
    build_dataset,
    make_splits,
    next_line_count,
    page_range,
    read_truth,
)
from datagen.consistency import check_po
from datagen.generate import generate_po
from datagen.render import lines_per_page
from datagen.templates import format_money

N = 3


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[BuiltPO]]:
    out = tmp_path_factory.mktemp("synthetic")
    try:
        pos = asyncio.run(build_dataset(N, seed=7, out=out, dev_size=1))
    except PlaywrightError as exc:
        pytest.skip(f"Chromium not available ({exc.message.splitlines()[0]})")
    return out, pos


# --- Built dataset ----------------------------------------------------------------------------


def test_pdfs_open_with_page_counts_in_range(built: tuple[Path, list[BuiltPO]]) -> None:
    out, pos = built
    assert len(pos) == N
    for b in pos:
        with pymupdf.open(out / "pdfs" / f"{b.po_id}.pdf") as doc:
            low, high = page_range(b.has_tc_page)
            assert low <= doc.page_count <= high
            assert doc.page_count == b.page_count
            width, height = doc[0].rect.width, doc[0].rect.height
            assert (round(width), round(height)) == (595, 842)  # A4 in points


def test_truth_files_validate_and_are_consistent(built: tuple[Path, list[BuiltPO]]) -> None:
    out, pos = built
    for b in pos:
        po, meta = read_truth(out / "truth" / f"{b.po_id}.json")
        assert po == b.po
        assert check_po(po) == []
        assert meta["layout"] == b.layout
        assert meta["page_count"] == b.page_count
        assert meta["seed"] == b.generated.meta.seed
        assert meta["knobs"]["min_lines"] <= meta["line_count"] <= meta["knobs"]["max_lines"]


def test_printed_text_matches_truth(built: tuple[Path, list[BuiltPO]]) -> None:
    out, pos = built
    for b in pos:
        with pymupdf.open(out / "pdfs" / f"{b.po_id}.pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        assert b.po.po_number in text
        assert format_money(b.po.grand_total) in text  # Indian format, e.g. 66,77,142.00
        assert b.po.vendor_gstin in text


def test_every_line_is_mapped_to_a_page_in_order(built: tuple[Path, list[BuiltPO]]) -> None:
    _, pos = built
    for b in pos:
        flat = [line_no for page in b.lines_per_page for line_no in page]
        assert flat == list(range(1, len(b.po.line_items) + 1))
        assert len(b.lines_per_page) == b.page_count
        assert b.lines_per_page[0], "first page must carry line items"


def test_layouts_round_robin_and_ids(built: tuple[Path, list[BuiltPO]]) -> None:
    _, pos = built
    assert [b.po_id for b in pos] == ["PO_0001", "PO_0002", "PO_0003"]
    assert [b.layout for b in pos] == LAYOUT_IDS[:N]


def test_manifest_splits_and_preview(built: tuple[Path, list[BuiltPO]]) -> None:
    out, pos = built
    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    assert [r["id"] for r in rows] == [b.po_id for b in pos]
    assert all((out / r["pdf"]).exists() and (out / r["truth"]).exists() for r in rows)
    splits = json.loads((out / "splits.json").read_text())
    assert sorted(splits["dev"] + splits["test"]) == [b.po_id for b in pos]
    assert len(splits["dev"]) == 1
    preview = (out / "preview.html").read_text()
    assert all(f"thumbs/{b.po_id}.png" in preview for b in pos)
    assert all((out / "thumbs" / f"{b.po_id}.png").exists() for b in pos)


# --- Pure logic (no browser) ------------------------------------------------------------------


def test_page_range_allows_extra_page_only_with_tc() -> None:
    assert page_range(False) == (2, 3)
    assert page_range(True) == (2, 4)


def test_next_line_count_moves_in_the_right_direction() -> None:
    assert next_line_count(60, 5, 2, 3) < 60  # too many pages -> fewer lines
    assert next_line_count(30, 1, 2, 3) > 30  # too few pages -> more lines
    assert next_line_count(40, 4, 2, 3) == 25  # scaled to the middle of the range
    assert next_line_count(9, 9, 2, 3) == 8  # never below the floor


def test_splits_are_stratified_partitioned_and_deterministic() -> None:
    layout_by_id = {f"PO_{i + 1:04d}": LAYOUT_IDS[i % 6] for i in range(60)}
    splits = make_splits(layout_by_id, dev_size=10, seed=42)
    assert len(splits["dev"]) == 10 and len(splits["test"]) == 50
    assert set(splits["dev"]).isdisjoint(splits["test"])
    dev_layouts = [layout_by_id[i] for i in splits["dev"]]
    assert set(dev_layouts) == set(LAYOUT_IDS)  # at least one per layout
    assert max(dev_layouts.count(x) for x in LAYOUT_IDS) <= 2  # stratified
    assert make_splits(layout_by_id, 10, 42) == splits
    assert make_splits(layout_by_id, 10, 43) != splits


def test_lines_per_page_from_page_texts() -> None:
    po = generate_po(3).po
    keys = [
        f"{format_money(i.taxable_value)} ... {format_money(i.line_total)}" for i in po.line_items
    ]
    half = len(keys) // 2
    # The grand-total page repeats nothing from the rows; a later page repeating an early
    # line's amounts must not move that line.
    texts = ["\n".join(keys[:half]), "\n".join(keys[half:]), keys[0]]
    mapping = lines_per_page(po, texts)
    assert mapping[0] == list(range(1, half + 1))
    assert mapping[1] == list(range(half + 1, len(keys) + 1))
    assert mapping[2] == []
