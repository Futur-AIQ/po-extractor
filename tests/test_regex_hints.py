"""Tests for regex hints (Step 3.2). No LLM, no network.

The dataset test reads data/synthetic (built by `make dataset`) and is skipped without it.
"""

import csv
import json
from pathlib import Path

import pymupdf
import pytest

from app.common.gst import gstin_check_char
from app.core.settings import Settings
from app.extraction.preprocess import prepare_sync
from app.extraction.regex_hints import extract_hints, hints_from_doc

DATASET = Path("data/synthetic")
needs_dataset = pytest.mark.skipif(
    not (DATASET / "manifest.csv").exists(), reason="data/synthetic not built (make dataset)"
)

VALID_GSTIN = "27AAPFU0939F1ZV"  # example from official GST documentation
BAD_CHECKSUM = VALID_GSTIN[:14] + "W"
BAD_STATE = "45AAPFU0939F1Z" + gstin_check_char("45AAPFU0939F1Z")  # 45 is not a state code


def make_pdf(*pages: str) -> bytes:
    """A native PDF with one A4 page per text."""
    with pymupdf.open() as doc:
        for text in pages:
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text, fontsize=9)
        return doc.tobytes()


# --- GSTIN and PAN ---------------------------------------------------------------------------


def test_gstins_split_into_valid_and_invalid_candidates() -> None:
    text = f"GSTIN: {VALID_GSTIN}\nVendor GSTIN {BAD_CHECKSUM}\nShip to {BAD_STATE}\n{VALID_GSTIN}"
    hints = extract_hints([text])
    assert hints.gstins == (VALID_GSTIN,)  # deduplicated
    assert hints.invalid_gstins == (BAD_CHECKSUM, BAD_STATE)


def test_gstin_candidates_are_uppercased_and_need_word_boundaries() -> None:
    hints = extract_hints([f"gstin {VALID_GSTIN.lower()} / X{VALID_GSTIN}9"])
    assert hints.gstins == (VALID_GSTIN,) and hints.invalid_gstins == ()


def test_pans_printed_alone_with_a_known_holder_type() -> None:
    text = f"PAN: AAPFU0939F\nGSTIN {VALID_GSTIN}\nRef ABCDE1234F\nPAN No. bbbpc1234k"
    # The PAN inside the GSTIN is not listed; ABCDE1234F has holder type 'D' (not a PAN).
    assert extract_hints([text]).pans == ("AAPFU0939F", "BBBPC1234K")


# --- Emails and phones -------------------------------------------------------------------


def test_emails_are_lowercased_and_deduplicated() -> None:
    text = "E-Mail : Purchase@Acme-Industries.co.in\nmail: purchase@acme-industries.co.in."
    assert extract_hints([text]).emails == ("purchase@acme-industries.co.in",)


@pytest.mark.parametrize(
    ("text", "phone"),
    [
        ("Contact : +919629010123", "+919629010123"),
        ("Ph: +91 98200 12345", "+919820012345"),
        ("Mob - 06894331324", "06894331324"),
        ("Tel - 022-23456789", "02223456789"),
        ("Phone 982 001 2345", "9820012345"),
        ("Contact Person\nPhone\n\n9005976744\n", "9005976744"),  # value on its own line
    ],
)
def test_phone_formats(text: str, phone: str) -> None:
    assert extract_hints([text]).phones == (phone,)


@pytest.mark.parametrize(
    "text",
    [
        "Grand total 1,234,567,890.00",  # an amount
        "26 15107003 Hex bolt M12",  # line number and an item code
        "Indent No. 4500080119 dated",  # a long number inside a sentence, no label
        "PO/2025-26/9820012345",  # part of a code
    ],
)
def test_numbers_that_are_not_phones(text: str) -> None:
    assert extract_hints([text]).phones == ()


# --- Dates and HSN/SAC -------------------------------------------------------------------


def test_dates_in_indian_formats_become_iso() -> None:
    text = (
        "PO Date: 14/03/2026  Delivery 15-03-2026  Validity 16.03.2026  Quotation 17.03.26\n"
        "Dated 18-Mar-2026, required by 19 Mar 2026, issued on 20th March 2026\n"
        "Amendment March 21, 2026; ISO 2026-03-22; Dt.23/03/2026"
    )
    assert extract_hints([text]).dates == tuple(f"2026-03-{day}" for day in range(14, 24))


def test_impossible_and_month_first_dates_are_dropped() -> None:
    text = "31/02/2026 03/14/2026 14 Mayor 2026 1 Declaration 2026 PO/2025-26/01"
    assert extract_hints([text]).dates == ()


def test_dates_are_deduplicated_in_order() -> None:
    assert extract_hints(["14-Mar-2026 01/04/2026 14/03/2026"]).dates == (
        "2026-03-14",
        "2026-04-01",
    )


def test_hsn_codes() -> None:
    text = (
        "1 Hex bolt\n731815\n2 Motor HSN: 8501 rpm 1440 IS 2062\n3 Service SAC 998719\n"
        "4 Item 84834000 2 NOS\nMangaluru - 575677\nPIN 400001\n1,03,378.00 003915\n"
    )
    assert extract_hints([text]).hsn_codes == ("731815", "998719", "84834000", "8501")


# --- From a prepared PDF -----------------------------------------------------------------


def test_hints_come_from_native_kept_pages_only() -> None:
    native = f"PURCHASE ORDER\nVendor GSTIN: {VALID_GSTIN}\nE-Mail: sales@vendor.in\n" + "x" * 60
    nearly_blank = "GSTIN 29ABCDE1234F1Z5"  # under 50 letters/digits: classified as scanned
    doc = prepare_sync(make_pdf(native, nearly_blank), Settings(_env_file=None))
    assert [page.kind for page in doc.pages] == ["native", "scanned"]
    hints = hints_from_doc(doc)
    assert hints.gstins == (VALID_GSTIN,) and hints.emails == ("sales@vendor.in",)


# --- Against the synthetic dataset -----------------------------------------------------------


@needs_dataset
def test_all_truth_identifiers_found_on_native_docs() -> None:
    settings = Settings(_env_file=None, native_mode="text")  # text only: no rendering needed
    with (DATASET / "manifest.csv").open(encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["variant"] == "N"]
    assert rows
    for row in rows:
        po = json.loads((DATASET / row["truth"]).read_text(encoding="utf-8"))["po"]
        hints = hints_from_doc(prepare_sync((DATASET / row["pdf"]).read_bytes(), settings))
        expected = {
            "gstins": ("buyer_gstin", "vendor_gstin", "bill_to_gstin", "ship_to_gstin"),
            "pans": ("buyer_pan", "vendor_pan"),
            "emails": ("buyer_email", "vendor_email"),
            "phones": ("buyer_phone", "vendor_phone"),
            "dates": ("po_date", "delivery_date", "po_validity_date", "quotation_date"),
        }
        for hint, fields in expected.items():
            values = {po[name] for name in fields if po.get(name)}
            assert values <= set(getattr(hints, hint)), (row["id"], hint)
        hsn = {line["hsn_sac"] for line in po["line_items"] if line.get("hsn_sac")}
        assert hsn <= set(hints.hsn_codes), row["id"]
        assert hints.invalid_gstins == (), row["id"]
