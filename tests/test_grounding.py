"""Tests for the grounding index (Step 3.2). No LLM, no network.

The dataset tests read data/synthetic (built by `make dataset`) and are skipped without it.
"""

import csv
import json
from pathlib import Path

import pymupdf
import pytest

from app.common.gst import gstin_check_char, is_valid_gstin
from app.core.settings import Settings
from app.extraction.grounding import GroundingIndex, normalise
from app.extraction.preprocess import prepare_sync

DATASET = Path("data/synthetic")
needs_dataset = pytest.mark.skipif(
    not (DATASET / "manifest.csv").exists(), reason="data/synthetic not built (make dataset)"
)
FAST = Settings(_env_file=None, image_dpi=24)  # tiny images: these tests only need text

PAGE = """PURCHASE ORDER
PO No: PO/2025-
26/00042   Date: 03/04/2026
Vendor GSTIN: 29ABCDE1234F1Z5
1 Hex bolt M12 x 50   HSN 7318   Qty 100 NOS   Rate 12.50   Amount 1,25,000.00
2 CMP-
7465 Compressor   HSN 841480
"""


def make_pdf(*pages: str) -> bytes:
    """A native PDF with one A4 page per text."""
    with pymupdf.open() as doc:
        for text in pages:
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text, fontsize=9)
        return doc.tobytes()


def fabricated_gstin(real: str) -> str:
    """A checksum-valid GSTIN with the same state code but a different PAN."""
    digits = f"{(int(real[7:11]) + 1) % 10000:04d}"
    first14 = real[:7] + digits + real[11:14]
    gstin = first14 + gstin_check_char(first14)
    assert is_valid_gstin(gstin) and gstin != real
    return gstin


def truth_identifiers(po: dict) -> list[str]:
    """The identifiers that §9.4 rule 9 grounds: GSTINs, PANs, PO number, item codes, HSN."""
    names = ("po_number", "buyer_gstin", "vendor_gstin", "bill_to_gstin", "ship_to_gstin")
    values = [po[name] for name in (*names, "buyer_pan", "vendor_pan") if po.get(name)]
    for line in po["line_items"]:
        values += [line[name] for name in ("item_code", "hsn_sac") if line.get(name)]
    return values


# --- Unit tests --------------------------------------------------------------------------


def test_normalise_removes_case_spaces_and_separators() -> None:
    assert normalise(" po/2025-26 / 00042 ") == "PO20252600042"
    assert normalise("₹ -- |") == ""


def test_values_wrapped_or_differently_separated_are_found() -> None:
    index = GroundingIndex([PAGE], complete=True)
    assert index.is_grounded("PO/2025-26/00042") == "yes"  # wrapped after "PO/2025-"
    assert index.is_grounded("PO 2025 26 00042") == "yes"
    assert index.is_grounded("29abcde1234f1z5") == "yes"
    assert index.is_grounded("CMP-7465") == "yes"  # short, wrapped at the hyphen
    assert index.is_grounded("7318") == "yes" and index.is_grounded("841480") == "yes"


def test_short_values_must_match_whole_tokens() -> None:
    index = GroundingIndex([PAGE], complete=True)
    assert index.contains("125000")  # "1,25,000.00" -> tokens 1 25 000 00
    assert not index.contains("2500")  # inside "1,25,000.00", not a token boundary
    assert not index.contains("8414")  # prefix of the HSN 841480
    assert index.is_grounded("8414") == "no"


def test_missing_value_is_no_only_when_every_page_has_text() -> None:
    assert GroundingIndex([PAGE], complete=True).is_grounded("27AAPFU0939F1ZV") == "no"
    partial = GroundingIndex([PAGE], complete=False)
    assert partial.is_grounded("27AAPFU0939F1ZV") == "unverifiable"
    assert partial.is_grounded("29ABCDE1234F1Z5") == "yes"


def test_value_without_letters_or_digits_is_unverifiable() -> None:
    assert GroundingIndex([PAGE], complete=True).is_grounded(" - / ") == "unverifiable"


def test_from_doc_native_scanned_and_text_free_modes() -> None:
    pdf = make_pdf(PAGE)
    native = GroundingIndex.from_doc(prepare_sync(pdf, FAST))
    assert native.complete and native.is_grounded("27AAPFU0939F1ZV") == "no"

    mixed = prepare_sync(make_pdf(PAGE, "Signature"), FAST)  # page 2 has no text layer
    assert [page.kind for page in mixed.pages] == ["native", "scanned"]
    index = GroundingIndex.from_doc(mixed)
    assert not index.complete and index.is_grounded("27AAPFU0939F1ZV") == "unverifiable"
    assert index.is_grounded("29ABCDE1234F1Z5") == "yes"

    images_only = Settings(_env_file=None, image_dpi=24, native_mode="image")
    index = GroundingIndex.from_doc(prepare_sync(pdf, images_only))
    assert index.is_grounded("29ABCDE1234F1Z5") == "unverifiable"  # text not used: never "no"


def test_skipped_fieldless_pages_do_not_make_a_doc_unverifiable() -> None:
    terms = (
        "Terms and Conditions\n1. Acceptance of this order is binding.\n"
        "2. Prices are firm.\n3. Disputes are subject to local courts.\n"
    )
    doc = prepare_sync(make_pdf(PAGE, terms), FAST)
    assert [page.skipped for page in doc.pages] == [False, True]
    assert GroundingIndex.from_doc(doc).is_grounded("27AAPFU0939F1ZV") == "no"


# --- Against the synthetic dataset -----------------------------------------------------------


def manifest() -> list[dict[str, str]]:
    with (DATASET / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@needs_dataset
def test_truth_identifiers_grounded_and_fabrications_caught_on_native_docs() -> None:
    rows = [row for row in manifest() if row["variant"] == "N"]
    assert rows
    for row in rows:
        po = json.loads((DATASET / row["truth"]).read_text(encoding="utf-8"))["po"]
        index = GroundingIndex.from_doc(prepare_sync((DATASET / row["pdf"]).read_bytes(), FAST))
        assert index.complete, row["id"]
        for value in truth_identifiers(po):
            assert index.is_grounded(value) == "yes", (row["id"], value)
        assert index.is_grounded(fabricated_gstin(po["vendor_gstin"])) == "no", row["id"]


@needs_dataset
def test_scanned_twins_and_mixed_docs_are_unverifiable() -> None:
    for row in manifest():
        if row["variant"] == "N":
            continue
        po = json.loads((DATASET / row["truth"]).read_text(encoding="utf-8"))["po"]
        index = GroundingIndex.from_doc(prepare_sync((DATASET / row["pdf"]).read_bytes(), FAST))
        assert not index.complete, row["id"]
        assert index.is_grounded(fabricated_gstin(po["vendor_gstin"])) == "unverifiable"
        if row["variant"] == "S":  # no text layer at all: nothing can be confirmed
            assert {index.is_grounded(v) for v in truth_identifiers(po)} == {"unverifiable"}
