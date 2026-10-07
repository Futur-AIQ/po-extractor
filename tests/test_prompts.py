"""Tests for the prefix-stable message builder (Step 3.3). No LLM, no network.

The dataset test reads data/synthetic (built by `make dataset`) and is skipped without it.
"""

import csv
import json
import re
from pathlib import Path

import pytest

from app.core.settings import Settings
from app.extraction.preprocess import PageContent, PreparedDoc, prepare_sync
from app.extraction.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    header_task,
    lines_task,
    supplier_hints,
    without_image_data,
)
from schema.fields import COMPUTED_FIELDS, HEADER_FIELDS_FOR_LLM
from schema.llm_keys import SHORT_KEYS
from schema.llm_schemas import line_item_columns_instruction

DATASET = Path("data/synthetic")
needs_dataset = pytest.mark.skipif(
    not (DATASET / "manifest.csv").exists(), reason="data/synthetic not built (make dataset)"
)


def page(page_no: int, kind: str, text: str | None, image: str | None, **kw) -> PageContent:
    return PageContent(
        page_no=page_no,
        kind=kind,
        text_md=text,
        image_b64=image,
        image_mime="image/jpeg" if image else None,
        width=10 if image else None,
        height=14 if image else None,
        **kw,
    )


# Page 1 native, 2 scanned, 3 skipped terms and conditions, 4 native.
DOC = PreparedDoc(
    doc_sha256="0" * 64,
    page_count=4,
    pages=[
        page(1, "native", "PURCHASE ORDER\nPO No: PO/2025-26/00042", "AAAA"),
        page(2, "scanned", None, "BBBB"),
        page(3, "native", None, None, skipped=True, skip_reason="terms only"),
        page(4, "native", "Grand Total 1,45,678.08", "CCCC"),
    ],
)
TASKS = [header_task(), lines_task(), lines_task(1), lines_task(2), lines_task(4)]


def user_parts(messages: list[dict]) -> list[dict]:
    assert [m["role"] for m in messages] == ["system", "user"]
    return messages[1]["content"]


def texts(messages: list[dict]) -> list[str]:
    return [p["text"] for p in user_parts(messages) if p["type"] == "text"]


# --- Prefix stability --------------------------------------------------------------------


def test_everything_but_the_task_block_is_byte_identical() -> None:
    built = [build_messages(DOC, task) for task in TASKS]
    prefixes = {json.dumps([m[0], user_parts(m)[:-1]], ensure_ascii=False).encode() for m in built}
    assert len(prefixes) == 1
    assert len({user_parts(m)[-1]["text"] for m in built}) == len(TASKS)  # tasks all differ
    assert all(m[0]["content"] == SYSTEM_PROMPT for m in built)


def test_building_twice_gives_identical_messages() -> None:
    assert build_messages(DOC, header_task()) == build_messages(DOC, header_task())


# --- Page content ------------------------------------------------------------------------


def test_page_order_label_image_then_text_and_skipped_pages_absent() -> None:
    parts = user_parts(build_messages(DOC, header_task()))[:-1]
    kinds = [p["type"] if p["type"] == "image_url" else p["text"].split("\n")[0] for p in parts]
    assert kinds == [
        "Page 1 of 4 (native)",
        "image_url",
        "--- TEXT LAYER, page 1 (exact characters printed on the page) ---",
        "Page 2 of 4 (scanned)",
        "image_url",  # scanned: no text layer block
        "Page 4 of 4 (native)",
        "image_url",
        "--- TEXT LAYER, page 4 (exact characters printed on the page) ---",
    ]
    assert parts[2]["text"].endswith("PO/2025-26/00042\n--- END TEXT LAYER, page 1 ---")
    assert not any("page 3" in t.lower() for t in texts(build_messages(DOC, lines_task())))


def test_images_are_data_urls_with_the_page_mime_type() -> None:
    images = [p for p in user_parts(build_messages(DOC, header_task())) if p["type"] == "image_url"]
    assert [p["image_url"]["url"] for p in images] == [
        "data:image/jpeg;base64,AAAA",
        "data:image/jpeg;base64,BBBB",
        "data:image/jpeg;base64,CCCC",
    ]


def test_text_only_native_page_has_no_image_part() -> None:
    doc = PreparedDoc("0" * 64, 1, [page(1, "native", "PURCHASE ORDER", None)])
    parts = user_parts(build_messages(doc, header_task()))
    assert [p["type"] for p in parts] == ["text", "text", "text"]  # label, text layer, task


# --- Task blocks -------------------------------------------------------------------------


def test_header_task_lists_every_short_key_once() -> None:
    task = user_parts(build_messages(DOC, header_task()))[-1]["text"]
    keys = re.findall(r"^([a-z_]+)(?: \(number\))?: ", task, re.M)
    assert keys == [SHORT_KEYS[name] for name in HEADER_FIELDS_FOR_LLM]
    assert "gr_total (number): " in task and "po_no: " in task


def test_line_tasks_give_column_order_and_rows_output() -> None:
    all_pages = user_parts(build_messages(DOC, lines_task()))[-1]["text"]
    assert line_item_columns_instruction() in all_pages and '{"rows": [...]}' in all_pages
    assert "ONE row" in all_pages
    one_page = user_parts(build_messages(DOC, lines_task(4)))[-1]["text"]
    assert line_item_columns_instruction() in one_page
    assert "line items of page 4 only" in one_page and "first appears on page 4" in one_page


@pytest.mark.parametrize("task", TASKS)
def test_no_computed_field_names_anywhere(task) -> None:
    text = json.dumps(build_messages(DOC, task))
    names = COMPUTED_FIELDS | {SHORT_KEYS[name] for name in COMPUTED_FIELDS if name in SHORT_KEYS}
    assert names >= {"total_tax", "cgst_amount", "sgst_amount", "igst_amount", "tax_tot"}
    assert not [name for name in names if name in text]


@pytest.mark.parametrize("page_no", [3, 0, 9])  # skipped, out of range
def test_per_page_task_must_name_a_page_sent_to_the_llm(page_no: int) -> None:
    with pytest.raises(ValueError, match=f"page {page_no} is not sent"):
        build_messages(DOC, lines_task(page_no))


def test_supplier_hints_placeholder_changes_nothing() -> None:
    assert supplier_hints("ISS14") == "" and supplier_hints(None) == ""
    assert build_messages(DOC, header_task("ISS14")) == build_messages(DOC, header_task())


def test_system_prompt_rules() -> None:
    assert PROMPT_VERSION
    for rule in ("Never calculate", "exactly as printed", "123456.50", "Consignee", "Invoice To"):
        assert rule in SYSTEM_PROMPT


def test_without_image_data_replaces_only_images() -> None:
    messages = build_messages(DOC, header_task())
    shown = without_image_data(messages)
    urls = [p["image_url"]["url"] for p in user_parts(shown) if p["type"] == "image_url"]
    assert urls == ["<image/jpeg, 27 chars>"] * 3
    assert texts(shown) == texts(messages)
    assert user_parts(messages)[1]["image_url"]["url"].endswith("AAAA")  # input unchanged


# --- Against the synthetic dataset -----------------------------------------------------------


@needs_dataset
def test_mixed_document_prefix_is_identical_across_tasks() -> None:
    with (DATASET / "manifest.csv").open(encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["variant"] == "M")
    doc = prepare_sync((DATASET / row["pdf"]).read_bytes(), Settings(_env_file=None, image_dpi=24))
    kept = doc.kept_pages
    assert {p.kind for p in kept} == {"native", "scanned"}
    tasks = [header_task(), lines_task(), *(lines_task(p.page_no) for p in kept)]
    built = [build_messages(doc, task) for task in tasks]
    assert len({json.dumps([m[0], user_parts(m)[:-1]]) for m in built}) == 1
    labels = [t for t in texts(built[0]) if t.startswith("Page ")]
    assert labels == [f"Page {p.page_no} of {doc.page_count} ({p.kind})" for p in kept]
    text_blocks = [t for t in texts(built[0]) if t.startswith("--- TEXT LAYER")]
    assert len(text_blocks) == sum(p.kind == "native" for p in kept)
