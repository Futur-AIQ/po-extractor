"""Tests for scanned twins and mixed POs (Step 1.6).

Pure tests build tiny PDFs with PyMuPDF. Dataset tests build 3 POs with headless Chromium
(skipped if it is not installed: `uv run playwright install chromium`).
"""

import asyncio
import csv
import hashlib
import io
import json
import random
from pathlib import Path

import pymupdf
import pytest
from PIL import Image
from playwright.async_api import Error as PlaywrightError

from datagen.build import build_dataset
from datagen.generate import Knobs
from datagen.scanify import (
    NATIVE,
    SCAN_DPI,
    SCANNED,
    build_variants,
    choose_mixed,
    page_kinds,
    scan_page_image,
    scan_pdf,
)


def _text_pdf(pages: int) -> bytes:
    with pymupdf.open() as doc:
        for number in range(1, pages + 1):
            page = doc.new_page(width=595, height=842)  # A4 in points
            page.insert_text((72, 100), f"PURCHASE ORDER page {number} PO/2026-27/00457")
        return doc.tobytes()


# --- Pure functions ----------------------------------------------------------------------------


def test_scanned_page_has_no_text_layer_and_keeps_page_size() -> None:
    scanned = scan_pdf(_text_pdf(2), {0, 1}, "t")
    with pymupdf.open(stream=scanned, filetype="pdf") as doc:
        assert doc.page_count == 2
        for page in doc:
            assert page.get_text().strip() == ""
            assert (round(page.rect.width), round(page.rect.height)) == (595, 842)
            assert len(page.get_images()) == 1


def test_mixed_scan_keeps_native_text_only_on_native_pages() -> None:
    scanned = scan_pdf(_text_pdf(3), {1}, "t")
    with pymupdf.open(stream=scanned, filetype="pdf") as doc:
        assert [bool(page.get_text().strip()) for page in doc] == [True, False, True]


def test_scan_image_is_200_dpi_jpeg_and_seeded() -> None:
    with pymupdf.open(stream=_text_pdf(1), filetype="pdf") as doc:
        first = scan_page_image(doc[0], random.Random("a"))
        assert first == scan_page_image(doc[0], random.Random("a"))  # same seed, same scan
        assert first != scan_page_image(doc[0], random.Random("b"))
    assert first[:3] == b"\xff\xd8\xff"  # JPEG
    width, _ = Image.open(io.BytesIO(first)).size
    assert abs(width - 595 * SCAN_DPI / 72) <= 2  # A4 width at 200 dpi ~ 1653 px


def test_scan_pdf_is_byte_identical_on_rerun() -> None:
    pdf = _text_pdf(2)
    assert scan_pdf(pdf, {0}, "k") == scan_pdf(pdf, {0}, "k")


def test_page_kinds() -> None:
    assert page_kinds(3, {1}) == [NATIVE, SCANNED, NATIVE]


def test_choose_mixed_spreads_layouts_and_keeps_pages_mixed() -> None:
    layouts = ["L1", "L2", "L3"]
    rows = [
        {"id": f"PO_{i:04d}", "layout": layouts[i % 3], "page_count": str(2 + i % 3)}
        for i in range(1, 31)
    ]
    test_ids = {r["id"] for r in rows[5:]}
    chosen = choose_mixed(rows, test_ids, 6)
    assert len(chosen) == 6 and set(chosen) <= test_ids
    by_id = {r["id"]: r for r in rows}
    assert sorted(by_id[i]["layout"] for i in chosen) == ["L1", "L1", "L2", "L2", "L3", "L3"]
    for po_id, pages in chosen.items():
        assert 1 <= len(pages) <= 2
        assert len(pages) < int(by_id[po_id]["page_count"])  # at least one page stays native
    assert choose_mixed(rows, test_ids, 6) == chosen  # deterministic


# --- Dataset variants -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    try:
        asyncio.run(build_dataset(3, seed=7, out=out, dev_size=1, knobs=Knobs(p_tc_page=0.5)))
    except PlaywrightError as exc:
        pytest.skip(f"Chromium not available ({exc.message.splitlines()[0]})")
    build_variants(out, mixed_count=2, workers=2)
    return out


def _rows(out: Path) -> list[dict[str, str]]:
    with (out / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_twins_have_no_text_and_same_page_count(dataset: Path) -> None:
    rows = {r["id"]: r for r in _rows(dataset)}
    twins = [r for r in rows.values() if r["variant"] == "S"]
    assert len(twins) == 3
    for twin in twins:
        native = rows[twin["base_id"]]
        with pymupdf.open(dataset / twin["pdf"]) as doc:
            assert doc.page_count == int(native["page_count"])
            assert all(page.get_text().strip() == "" for page in doc)
        assert twin["page_kinds"] == " ".join([SCANNED] * int(native["page_count"]))
        assert twin["truth"] == native["truth"]  # same ground truth


def test_mixed_pdfs_match_recorded_page_kinds(dataset: Path) -> None:
    mixed = [r for r in _rows(dataset) if r["variant"] == "M"]
    assert len(mixed) == 2 and all(r["split"] == "test" for r in mixed)
    for row in mixed:
        kinds = row["page_kinds"].split()
        assert NATIVE in kinds and SCANNED in kinds
        with pymupdf.open(dataset / row["pdf"]) as doc:
            has_text = [bool(page.get_text().strip()) for page in doc]
        assert has_text == [kind == NATIVE for kind in kinds]


def test_truth_lists_variants(dataset: Path) -> None:
    for row in _rows(dataset):
        truth = json.loads((dataset / row["truth"]).read_text())
        variant = truth["meta"]["variants"][row["variant"]]
        assert variant["pdf"] == row["pdf"]
        assert variant["page_kinds"] == row["page_kinds"].split()
        assert (dataset / variant["pdf"]).exists()


def test_splits_list_every_variant(dataset: Path) -> None:
    splits = json.loads((dataset / "splits.json").read_text())
    rows = _rows(dataset)
    listed = {po_id for split in splits.values() for ids in split.values() for po_id in ids}
    assert listed == {r["id"] for r in rows}
    for name, split in splits.items():
        assert [f"{i}_S" for i in split["N"]] == split["S"]
        assert all(r["split"] == name for r in rows if r["id"] in split["M"])


def test_rerun_is_idempotent(dataset: Path) -> None:
    def snapshot() -> str:
        files = [
            *sorted(dataset.glob("pdfs/*.pdf")),
            *sorted(dataset.glob("truth/*.json")),
            dataset / "manifest.csv",
            dataset / "splits.json",
        ]
        return hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()

    before = snapshot()
    build_variants(dataset, mixed_count=2, workers=2)
    assert snapshot() == before
    assert len(_rows(dataset)) == 3 + 3 + 2  # no duplicated variant rows
