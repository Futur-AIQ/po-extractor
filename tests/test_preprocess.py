"""Tests for pre-processing (Step 3.1). No LLM, no network.

The dataset tests read data/synthetic (built by `make dataset`) and are skipped without it.
"""

import asyncio
import base64
import csv
import hashlib
import json
from pathlib import Path

import pymupdf
import pytest

from app.core.settings import Settings
from app.extraction.preprocess import (
    PreparedDoc,
    classify,
    fieldless_reason,
    meaningful_chars,
    prepare,
    prepare_sync,
)

DATASET = Path("data/synthetic")
needs_dataset = pytest.mark.skipif(
    not (DATASET / "manifest.csv").exists(), reason="data/synthetic not built (make dataset)"
)

TC_PAGE = """General Terms and Conditions of Purchase
1. Acceptance. The supplier shall return the duplicate copy of this order duly signed.
2. Price. Prices are firm and not subject to any escalation till completion of supply.
3. Inspection. Material is subject to inspection at our stores.
4. Jurisdiction. Disputes are subject to the courts at the purchaser's registered office.
"""
FIRST_PAGE = """PURCHASE ORDER
PO No: PO/2025-26/00042    Date: 03/04/2026
Vendor GSTIN: 29ABCDE1234F1Z5
1 Hex bolt M12 x 50     HSN 731815   Qty 100 NOS   Rate 12.50   Amount 1,250.00
"""


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def make_pdf(*pages: str) -> bytes:
    """A native PDF with one A4 page per text."""
    with pymupdf.open() as doc:
        for text in pages:
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text, fontsize=9)
        return doc.tobytes()


def manifest() -> list[dict[str, str]]:
    with (DATASET / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- Classification and field-less pages -------------------------------------------------


def test_meaningful_chars_ignores_spaces_punctuation_and_rules() -> None:
    assert meaningful_chars("ab 12 -- | . ₹\n") == 4


def test_classify_threshold_is_strictly_more_than_min_chars() -> None:
    assert classify("a" * 50, native_min_chars=50) == "scanned"
    assert classify("a" * 51 + " ---- " * 20, native_min_chars=50) == "native"
    assert classify("", native_min_chars=50) == "scanned"


def test_plain_terms_page_is_fieldless() -> None:
    assert fieldless_reason(TC_PAGE) is not None


@pytest.mark.parametrize(
    "extra",
    [
        "Invoices must quote our GSTIN 29ABCDE1234F1Z5.",
        "5. Payment: 30 days from receipt of invoice.",
        "6. Freight: extra at actuals.",
        "Material shall be delivered on or before 30/04/2026.",
        "Validity: up to 18-Feb-2027.",
        "Liquidated damages capped at Rs. 50,000.",
        "Sub total 1,23,456.00  Grand total 1,45,678.08",
        "Sl  Description  HSN  Qty  Rate  Amount",
    ],
)
def test_terms_page_with_any_field_is_kept(extra: str) -> None:
    assert fieldless_reason(TC_PAGE + extra + "\n") is None


def test_page_without_terms_heading_or_clauses_is_kept() -> None:
    assert fieldless_reason(TC_PAGE.replace("Terms and Conditions", "Notes")) is None
    assert fieldless_reason("Terms and Conditions\nAs per our standard terms.\n") is None


def test_page_one_is_never_skipped_and_skipping_can_be_disabled() -> None:
    pdf = make_pdf(TC_PAGE, FIRST_PAGE, TC_PAGE)
    doc = prepare_sync(pdf, settings())
    assert [p.skipped for p in doc.pages] == [False, False, True]
    assert doc.pages[2].skip_reason and doc.pages[2].image_b64 is None
    assert doc.pages[2].text_md is None and [p.page_no for p in doc.kept_pages] == [1, 2]
    assert not any(p.skipped for p in prepare_sync(pdf, settings(skip_fieldless_pages=False)).pages)


# --- Text, images and modes ----------------------------------------------------------------


def test_native_page_has_text_and_image_by_default() -> None:
    page = prepare_sync(make_pdf(FIRST_PAGE), settings()).pages[0]
    assert page.kind == "native" and "29ABCDE1234F1Z5" in page.text_md
    assert page.image_mime == "image/jpeg"
    assert base64.b64decode(page.image_b64)[:3] == b"\xff\xd8\xff"  # JPEG


def test_native_modes_for_ablations() -> None:
    pdf = make_pdf(FIRST_PAGE)
    text_only = prepare_sync(pdf, settings(native_mode="text")).pages[0]
    assert text_only.text_md and text_only.image_b64 is None and text_only.width is None
    image_only = prepare_sync(pdf, settings(native_mode="image")).pages[0]
    assert image_only.text_md is None and image_only.image_b64
    assert image_only.kind == "native"


def test_png_option() -> None:
    page = prepare_sync(make_pdf(FIRST_PAGE), settings(image_format="png")).pages[0]
    assert page.image_mime == "image/png"
    assert base64.b64decode(page.image_b64)[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("dpi", [72, 120, 150, 200])
def test_image_size_follows_dpi(dpi: int) -> None:
    page = prepare_sync(make_pdf(FIRST_PAGE), settings(image_dpi=dpi)).pages[0]
    assert abs(page.width - 595 * dpi / 72) <= 1 and abs(page.height - 842 * dpi / 72) <= 1


def test_longest_side_is_capped() -> None:
    page = prepare_sync(make_pdf(FIRST_PAGE), settings(image_dpi=600, max_image_px=1000)).pages[0]
    assert 999 <= page.height <= 1000 and page.width < page.height
    assert abs(page.width / page.height - 595 / 842) < 0.01


def test_pymupdf4llm_engine_gives_markdown_text() -> None:
    page = prepare_sync(make_pdf(FIRST_PAGE), settings(text_engine="pymupdf4llm")).pages[0]
    assert "29ABCDE1234F1Z5" in page.text_md and page.image_b64


def test_sha256_and_timings_recorded() -> None:
    pdf = make_pdf(FIRST_PAGE, TC_PAGE)
    doc = prepare_sync(pdf, settings())
    assert doc.doc_sha256 == hashlib.sha256(pdf).hexdigest() and doc.page_count == 2
    assert set(doc.timings_ms) == {"classify", "text", "render", "total"}
    assert doc.timings_ms["total"] > 0
    assert doc.timings_ms["total"] >= max(v for k, v in doc.timings_ms.items() if k != "total")


def test_not_a_pdf_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not a readable PDF"):
        prepare_sync(b"hello", settings())


async def test_prepare_runs_off_the_event_loop() -> None:
    pdf = make_pdf(FIRST_PAGE, TC_PAGE)
    docs = await asyncio.gather(prepare(pdf, settings()), prepare(pdf, settings()))
    assert all(isinstance(doc, PreparedDoc) for doc in docs)
    assert docs[0].pages == docs[1].pages == prepare_sync(pdf, settings()).pages


# --- Against the synthetic dataset -----------------------------------------------------------


@needs_dataset
def test_classification_and_skipping_match_truth_for_all_variants() -> None:
    fast = settings(image_dpi=24)  # small images: this test is about kinds and skips
    rows = manifest()
    assert {row["variant"] for row in rows} == {"N", "S", "M"}
    for row in rows:
        meta = json.loads((DATASET / row["truth"]).read_text(encoding="utf-8"))["meta"]
        doc = prepare_sync((DATASET / row["pdf"]).read_bytes(), fast)
        kinds = row["page_kinds"].split()
        assert [page.kind for page in doc.pages] == kinds, row["id"]
        # Field-less pages are skipped only when native; no other page is ever skipped.
        expected = {n for n in meta["fieldless_pages"] if kinds[n - 1] == "native"}
        assert {page.page_no for page in doc.pages if page.skipped} == expected, row["id"]
        for page in doc.kept_pages:
            assert page.image_b64, row["id"]
            assert (page.text_md is not None) == (page.kind == "native"), row["id"]


@needs_dataset
def test_native_and_scanned_twins_have_the_same_image_size() -> None:
    row = next(r for r in manifest() if r["variant"] == "N")
    native = prepare_sync((DATASET / row["pdf"]).read_bytes(), settings())
    scanned = prepare_sync((DATASET / f"pdfs/{row['id']}_S.pdf").read_bytes(), settings())
    assert {p.kind for p in native.pages} == {"native"}
    assert {p.kind for p in scanned.pages} == {"scanned"}
    with pymupdf.open(DATASET / row["pdf"]) as pdf:
        expected = [(round(p.rect.width * 150 / 72), round(p.rect.height * 150 / 72)) for p in pdf]
    for doc in (native, scanned):
        for page in doc.kept_pages:
            assert abs(page.width - expected[page.page_no - 1][0]) <= 1
            assert abs(page.height - expected[page.page_no - 1][1]) <= 1
