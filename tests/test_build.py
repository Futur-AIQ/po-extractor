"""Tests for PDF rendering and the dataset build (Step 1.5, PRD v1.1 §11.1).

The build tests launch headless Chromium (local, no network). They are skipped if the
browser is not installed: run `uv run playwright install chromium`.
"""

import asyncio
import csv
import json
from collections import Counter
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
    rebalance_layouts,
    regenerate_with_lines,
)
from datagen.consistency import check_po
from datagen.generate import Knobs, generate_many, generate_po
from datagen.render import fieldless_pages, lines_per_page
from datagen.templates import format_money

N = 3


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[BuiltPO]]:
    out = tmp_path_factory.mktemp("synthetic")
    # A T&C page on every PO, so fieldless-page detection is exercised.
    knobs = Knobs(p_tc_page=1.0, duplicate_po_count=1)
    try:
        pos = asyncio.run(build_dataset(N, seed=7, out=out, dev_size=1, knobs=knobs))
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
            assert (round(doc[0].rect.width), round(doc[0].rect.height)) == (595, 842)  # A4


def test_truth_passes_consistency_and_has_v11_meta(built: tuple[Path, list[BuiltPO]]) -> None:
    out, pos = built
    for b in pos:
        po, meta = read_truth(out / "truth" / f"{b.po_id}.json")
        assert po == b.po
        assert check_po(po) == []
        assert meta["issuer_id"] == b.generated.meta.issuer_id
        assert meta["issuer_frequent"] == b.generated.meta.issuer_frequent
        assert meta["expected_duplicate"] == b.generated.meta.expected_duplicate
        assert meta["layout"] == b.generated.issuer.layout  # layout comes from the issuer
        assert meta["seed"] == b.generated.meta.seed
        assert meta["page_count"] == b.page_count
        assert meta["fieldless_pages"] == b.fieldless_pages
        assert meta["variants"] == {
            "N": {"pdf": f"pdfs/{b.po_id}.pdf", "page_kinds": ["native"] * b.page_count}
        }
        assert meta["knobs"]["min_lines"] <= meta["line_count"] <= meta["knobs"]["max_lines"]
    assert sum(b.generated.meta.expected_duplicate for b in pos) == 1


def test_printed_text_matches_truth(built: tuple[Path, list[BuiltPO]]) -> None:
    out, pos = built
    for b in pos:
        with pymupdf.open(out / "pdfs" / f"{b.po_id}.pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        assert b.po.po_number in text
        assert format_money(b.po.grand_total) in text  # Indian format, e.g. 66,77,142.00
        assert b.po.vendor_gstin in text


def test_lines_and_fieldless_pages(built: tuple[Path, list[BuiltPO]]) -> None:
    out, pos = built
    for b in pos:
        flat = [line_no for page in b.lines_per_page for line_no in page]
        assert flat == list(range(1, len(b.po.line_items) + 1))
        assert len(b.lines_per_page) == b.page_count
        assert b.lines_per_page[0], "first page must carry line items"
        assert 1 not in b.fieldless_pages
        with pymupdf.open(out / "pdfs" / f"{b.po_id}.pdf") as doc:
            for page in b.fieldless_pages:
                assert not b.lines_per_page[page - 1]
                assert "Terms" in doc[page - 1].get_text()
    # The generic T&C page has no fields; L6 prints the commercial terms on it, so it has.
    generic = [b for b in pos if b.layout != "L6_psu_formal"]
    assert all(b.page_count in b.fieldless_pages for b in generic)  # the appended T&C page


def test_masters_manifest_splits_preview(built: tuple[Path, list[BuiltPO]]) -> None:
    out, pos = built
    parties = {p["gstin"] for p in json.loads((out / "masters" / "parties.json").read_text())}
    assert {b.po.vendor_gstin for b in pos} | {b.po.buyer_gstin for b in pos} <= parties
    processed = json.loads((out / "masters" / "processed_po_numbers.json").read_text())
    for b in pos:
        numbers = processed.get(b.po.buyer_gstin, {}).get("po_numbers", [])
        assert (b.po.po_number in numbers) == b.generated.meta.expected_duplicate

    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    assert [r["id"] for r in rows] == ["PO_0001", "PO_0002", "PO_0003"]
    assert all((out / r["pdf"]).exists() and (out / r["truth"]).exists() for r in rows)
    assert [r["layout"] for r in rows] == [b.layout for b in pos]
    assert all(r["variant"] == "N" and r["base_id"] == r["id"] for r in rows)
    splits = json.loads((out / "splits.json").read_text())
    assert sorted(splits["dev"]["N"] + splits["test"]["N"]) == [b.po_id for b in pos]
    preview = (out / "preview.html").read_text()
    assert all(f"thumbs/{b.po_id}.png" in preview for b in pos)


# --- Pure logic (no browser) ------------------------------------------------------------------


def test_page_range_allows_fifth_page_only_with_tc() -> None:
    assert page_range(False) == (2, 4)
    assert page_range(True) == (2, 5)


def test_next_line_count_moves_in_the_right_direction() -> None:
    assert next_line_count(60, 6, 2, 4) < 60  # too many pages -> fewer lines
    assert next_line_count(30, 1, 2, 4) > 30  # too few pages -> more lines
    assert next_line_count(40, 6, 2, 4) == 20  # scaled to the middle of the range (3 pages)
    assert next_line_count(21, 9, 2, 4) == 20  # never below the PRD minimum
    assert next_line_count(79, 1, 2, 4) == 80  # never above the PRD maximum


def test_rebalance_gives_every_layout_six_and_keeps_frequent_issuers() -> None:
    pos = generate_many(60, seed=3)
    balanced = rebalance_layouts(pos, 6)
    counts = Counter(g.meta.layout for g in balanced)
    assert all(counts[layout] >= 6 for layout in LAYOUT_IDS)
    for before, after in zip(pos, balanced, strict=True):
        assert after.issuer.layout == after.meta.layout
        if before.meta.issuer_frequent:
            assert after == before  # frequent issuers keep their fixed layout
        else:
            assert after.po == before.po  # only the layout may move
    assert rebalance_layouts(pos, 6) == balanced  # deterministic


def test_regenerate_keeps_issuer_po_number_and_dataset_flags() -> None:
    planned = generate_many(30, seed=4)
    duplicate = next(g for g in planned if g.meta.expected_duplicate)
    again = regenerate_with_lines(duplicate, Knobs(min_lines=25, max_lines=25))
    assert len(again.po.line_items) == 25
    assert again.po.po_number == duplicate.po.po_number
    assert again.meta.issuer_id == duplicate.meta.issuer_id
    assert again.meta.expected_duplicate and again.meta.layout == duplicate.meta.layout
    assert check_po(again.po) == []


def test_splits_are_stratified_partitioned_and_deterministic() -> None:
    layout_by_id = {f"PO_{i + 1:04d}": LAYOUT_IDS[i % 6] for i in range(60)}
    splits = make_splits(layout_by_id, dev_size=10, seed=42)
    assert len(splits["dev"]) == 10 and len(splits["test"]) == 50
    assert set(splits["dev"]).isdisjoint(splits["test"])
    dev_layouts = [layout_by_id[i] for i in splits["dev"]]
    assert set(dev_layouts) == set(LAYOUT_IDS)  # at least one per layout
    assert max(dev_layouts.count(x) for x in LAYOUT_IDS) <= 2  # stratified
    assert make_splits(layout_by_id, 10, 42) == splits


def test_lines_per_page_from_page_texts() -> None:
    po = generate_po(3).po
    keys = [
        f"{format_money(i.taxable_value)} ... {format_money(i.line_total)}" for i in po.line_items
    ]
    half = len(keys) // 2
    # A later page repeating an early line's amounts must not move that line.
    texts = ["\n".join(keys[:half]), "\n".join(keys[half:]), keys[0]]
    mapping = lines_per_page(po, texts)
    assert mapping[0] == list(range(1, half + 1))
    assert mapping[1] == list(range(half + 1, len(keys) + 1))
    assert mapping[2] == []


def test_fieldless_pages_ignore_running_header() -> None:
    po = generate_po(5, Knobs(optional_drop_rate=0)).po
    furniture = f"{po.buyer_name}\nPurchase Order {po.po_number}\nPage 3 of 3\n"
    texts = [
        furniture + po.vendor_gstin,  # a header field: not fieldless
        furniture + "rows",  # carries line items (see mapping): not fieldless
        furniture + "1. Acceptance. The supplier shall ...",  # only boilerplate: fieldless
    ]
    mapping = [[], [1], []]
    assert fieldless_pages(po, texts, "L1_classic_erp", mapping) == [3]
