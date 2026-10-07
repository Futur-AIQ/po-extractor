"""Prompts and messages for extraction calls (PRD §6.3, §9.2, §9.3).

Every call for one PO has the same prefix, so vLLM prefix caching reads the pages once:

    system prompt  ->  page content (per page: label, image, text layer if native)  ->  task

Only the final task block differs between calls: the header fields, all line items, or the
line items of one page. The page content depends only on the PreparedDoc (native_mode has
already decided there whether a native page carries its image, its text layer or both),
so it is byte-identical for every call of the same document.

Dates: the prompt asks for dates exactly as printed, not YYYY-MM-DD. Asked to reformat, a
small model copied digits in print order ("14/03/2026" -> "1403-20-26"); code parses the
printed date day-first instead (schema/dates.py).
"""

from dataclasses import dataclass
from typing import Any, Literal

from app.extraction.preprocess import PageContent, PreparedDoc
from schema.llm_schemas import header_schema_for_llm, line_item_columns_instruction

PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = """\
You are a precise document data extractor for Indian GST purchase orders (POs).
You receive the pages of one PO. Each page has an image; native pages also have their text \
layer, which holds the exact characters printed on the page. Use the image to see the layout \
(which value belongs to which label, column or party) and the text layer to copy characters \
exactly.

Rules:
1. Extract only values printed on the document. Never guess or infer a value.
2. Never calculate. Do not add, multiply or derive totals, taxes or amounts, even when a \
printed value seems to be missing.
3. Omit a field that is not printed: no null, "", "N/A" or notes. (In line-item rows, use \
null for a cell that is not printed, because every row has a fixed number of cells.)
4. Dates: copy exactly as printed, e.g. "14/03/2026" or "14-Mar-2026". Do not reformat them.
5. Numbers: plain JSON numbers without currency symbols, units or thousands separators. \
Indian grouping "1,23,456.50" -> 123456.50; "18 %" -> 18; a deduction "(-) 0.24" -> -0.24.
6. Identifiers (GSTIN, PAN, PO number, item codes, HSN/SAC, phone numbers, emails): copy \
character by character, exactly as printed, keeping "/", "-" and leading zeros.
7. Text: copy as printed; join wrapped lines with a space (address lines with ", ").
8. Label synonyms: Supplier, Seller or M/s = vendor; Consignee, Deliver To or Ship To = \
ship-to; Invoice To or Bill To = bill-to; Basic Amount or Basic Value = taxable value; \
Purchaser = buyer (the company issuing the PO).
9. Return only the JSON object, nothing else."""


@dataclass(frozen=True)
class Task:
    """What one call extracts: the header, or the line items of all pages or of one page."""

    kind: Literal["header", "lines"]
    page_no: int | None = None  # lines only: rows that start on this page
    issuer_id: str | None = None  # for supplier_hints


def header_task(issuer_id: str | None = None) -> Task:
    """All header fields (one call)."""
    return Task("header", issuer_id=issuer_id)


def lines_task(page_no: int | None = None, issuer_id: str | None = None) -> Task:
    """Line items of every page, or only those starting on `page_no`."""
    return Task("lines", page_no=page_no, issuer_id=issuer_id)


# =========================================================================================
# Page content (the shared prefix)
# =========================================================================================


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image(page: PageContent) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{page.image_mime};base64,{page.image_b64}"},
    }


def page_parts(doc: PreparedDoc) -> list[dict[str, Any]]:
    """Content parts for every page sent to the LLM: label, image, then text layer if any.

    Page numbers are the PDF's own (skipped field-less pages leave a gap), so the labels match
    the "Page x of y" printed on the PO.
    """
    parts = []
    for page in doc.kept_pages:
        parts.append(_text(f"Page {page.page_no} of {doc.page_count} ({page.kind})"))
        if page.image_b64:
            parts.append(_image(page))
        if page.text_md:
            parts.append(
                _text(
                    f"--- TEXT LAYER, page {page.page_no} "
                    "(exact characters printed on the page) ---\n"
                    f"{page.text_md}\n"
                    f"--- END TEXT LAYER, page {page.page_no} ---"
                )
            )
    return parts


# =========================================================================================
# Task blocks (last, different for every call)
# =========================================================================================


def supplier_hints(issuer_id: str | None) -> str:
    """Layout notes for a known issuer. Placeholder for a phase-2 feature: always ""."""
    return ""


def header_task_text() -> str:
    """Header task: one JSON object with the short keys below, absent fields omitted."""
    lines = [
        "TASK: Extract the purchase order header fields from the pages above.",
        "Return one JSON object that uses only the keys below. Omit a key when its field is "
        "not printed on the document.",
        "",
        "Keys (key: meaning):",
    ]
    for key, prop in header_schema_for_llm()["properties"].items():
        kind = " (number)" if prop["type"] == "number" else ""
        lines.append(f"{key}{kind}: {prop['description']}")
    return "\n".join(lines)


def lines_task_text(page_no: int | None = None) -> str:
    """Line-item task for all pages, or only the rows that start on `page_no`."""
    if page_no is None:
        scope = [
            "TASK: Extract every line item from the item table on the pages above.",
            "Output one row per printed line item, in printed order. A row split across a page "
            "break is ONE row: join its parts and do not repeat it.",
        ]
    else:
        scope = [
            f"TASK: Extract the line items of page {page_no} only.",
            f"Output a row only if its row number first appears on page {page_no}, in printed "
            f"order. If a row starts on page {page_no} and continues on the next page, output it "
            "once, complete, with its parts joined. Do not output a row that continues from the "
            "previous page.",
        ]
    return "\n".join(
        [
            *scope,
            "Do not output subtotal, tax summary or carried-forward rows.",
            'Return {"rows": [...]}.',
            "",
            line_item_columns_instruction(),
        ]
    )


def task_text(task: Task) -> str:
    """The final task block, with supplier hints (if any) inside it."""
    text = header_task_text() if task.kind == "header" else lines_task_text(task.page_no)
    hints = supplier_hints(task.issuer_id)
    return f"{text}\n\n{hints}" if hints else text


# =========================================================================================
# Messages
# =========================================================================================


def build_messages(doc: PreparedDoc, task: Task) -> list[dict[str, Any]]:
    """OpenAI-format chat messages: system prompt, then pages, then the task (last part).

    Raises ValueError if a per-page task names a page that is not sent to the LLM.
    """
    if task.page_no is not None and task.page_no not in {p.page_no for p in doc.kept_pages}:
        raise ValueError(f"page {task.page_no} is not sent to the LLM (skipped or out of range)")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [*page_parts(doc), _text(task_text(task))]},
    ]


def without_image_data(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of `messages` with each image replaced by "<mime, N chars>" (for logs and traces)."""
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = [
                _image_placeholder(part) if part["type"] == "image_url" else part
                for part in content
            ]
        out.append({**message, "content": content})
    return out


def _image_placeholder(part: dict[str, Any]) -> dict[str, Any]:
    url = part["image_url"]["url"]
    mime = url[len("data:") : url.index(";")]
    return {"type": "image_url", "image_url": {"url": f"<{mime}, {len(url)} chars>"}}
