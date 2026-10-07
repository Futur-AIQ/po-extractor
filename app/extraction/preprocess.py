"""Pre-processing: turn PDF bytes into per-page content for the LLM (PRD §5, FR-04, FR-05, §9.1).

For every page:
1. Classify it, per page, so one PDF can mix both kinds: native when its text layer has more
   than `native_min_chars` letters/digits, otherwise scanned.
2. Skip it if it is confidently field-less (a terms-and-conditions page that carries no PO
   field). Only native pages other than page 1 are ever skipped.
3. Native pages get their text layer and a page image; scanned pages get only an image.
   `native_mode` can drop the image or the text of native pages for ablations.

Text engine (`text_engine`): the default is PyMuPDF plain text in reading order. On the
synthetic set it keeps every printed field value and takes ~10 ms per PO. pymupdf4llm
markdown (legacy mode, as layout mode splits words across table cells) is kept for
ablations. It repeats merged table cells, so its text is 2-4x longer, and it costs up to
~4 s of CPU per PO.

PyMuPDF is not thread-safe, so all PDF work runs under one lock in a worker thread;
`prepare` keeps it off the event loop.
"""

import asyncio
import base64
import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import pymupdf

from app.core.settings import Settings

PageKind = Literal["native", "scanned"]

_PDF_LOCK = threading.Lock()
_POINTS_PER_INCH = 72


@dataclass(frozen=True)
class PageContent:
    """One page as the LLM will see it.

    `text_md` is the page's text layer (native pages only): plain text with the default
    engine, markdown with pymupdf4llm. The image fields are None when no image is sent: for
    skipped pages, and for native pages when native_mode is "text".
    """

    page_no: int  # 1-based
    kind: PageKind
    text_md: str | None
    image_b64: str | None
    image_mime: str | None
    width: int | None  # image size in pixels
    height: int | None
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True)
class PreparedDoc:
    """A PDF ready for the call planner: one PageContent per page, skipped ones included."""

    doc_sha256: str
    page_count: int
    pages: list[PageContent]
    timings_ms: dict[str, float] = field(default_factory=dict)  # classify, text, render, total

    @property
    def kept_pages(self) -> list[PageContent]:
        """The pages sent to the LLM."""
        return [page for page in self.pages if not page.skipped]


# =========================================================================================
# Classification and field-less pages
# =========================================================================================


def meaningful_chars(text: str) -> int:
    """Number of letters and digits in `text` (spaces, punctuation and rules don't count)."""
    return sum(ch.isalnum() for ch in text)


def classify(text: str, native_min_chars: int) -> PageKind:
    """FR-04: native if the text layer has more than `native_min_chars` letters/digits."""
    return "native" if meaningful_chars(text) > native_min_chars else "scanned"


_TC_HEADING = re.compile(r"^.{0,25}\bterms\s*(?:&|and)\s*conditions\b.{0,20}$", re.I | re.M)
_NUMBERED_CLAUSE = re.compile(r"^\s*(?:\d{1,2}|[a-z]|[ivx]{1,4})[.)]\s+\S", re.I | re.M)
MIN_CLAUSES = 3

# Anything below means the page may carry a PO field, so it is kept.
_BLOCKERS = {
    "GSTIN": re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b"),
    "currency amount": re.compile(r"(?:₹|\bRs\.?|\bINR)\s*\d"),
    "date": re.compile(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
        r"|\b\d{1,2}[- ](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-, ]+\d{2,4}\b",
        re.I,
    ),
    # Labelled commercial terms ("Payment: 30 days", "Freight: extra") are header fields.
    "labelled commercial term": re.compile(
        r"^\s*(?:\d{1,2}[.)]\s*)?(?:payment|price basis|delivery|freight|mode of|despatch"
        r"|dispatch|transport|place of delivery|packing|warranty|guarantee|validity)"
        r"\b[^:\n]{0,30}:",
        re.I | re.M,
    ),
}
_MONEY = re.compile(r"\b\d{1,3}(?:,\d{2,3})*\.\d{2}\b")  # 1,23,456.00 / 404.00
_QTY_HEADER = re.compile(r"\b(?:qty|quantity)\b", re.I)
_ITEM_HEADER = re.compile(r"\b(?:hsn|rate|uom|unit price)\b", re.I)


def fieldless_reason(text: str) -> str | None:
    """Why a native page can be skipped, or None to keep it.

    Skips only when confident: a terms-and-conditions heading and numbered clauses, and none
    of a GSTIN, a date, a labelled commercial term, currency amounts or a line-item table.
    """
    if not _TC_HEADING.search(text) or len(_NUMBERED_CLAUSE.findall(text)) < MIN_CLAUSES:
        return None
    if any(pattern.search(text) for pattern in _BLOCKERS.values()):
        return None
    if len(_MONEY.findall(text)) >= 2:  # an amounts table
        return None
    if _QTY_HEADER.search(text) and _ITEM_HEADER.search(text):  # a line-item table
        return None
    return "terms and conditions only (no GSTIN, dates, terms, amounts or line items)"


# =========================================================================================
# Text and images
# =========================================================================================


def _markdown_pages(doc: pymupdf.Document, page_indexes: list[int]) -> dict[int, str]:
    """pymupdf4llm markdown per page (0-based index), legacy mode, no OCR."""
    import pymupdf4llm  # imported only for this engine: its import activates layout mode

    pymupdf4llm.use_layout(False)
    chunks = pymupdf4llm.to_markdown(doc, pages=page_indexes, page_chunks=True, show_progress=False)
    return {index: chunk["text"].strip() for index, chunk in zip(page_indexes, chunks, strict=True)}


def render_page(
    page: pymupdf.Page, dpi: int, image_format: str, jpeg_quality: int, max_px: int
) -> tuple[bytes, int, int]:
    """(image bytes, width, height) at `dpi`, scaled down so the longest side is <= max_px."""
    zoom = dpi / _POINTS_PER_INCH
    longest = max(page.rect.width, page.rect.height) * zoom
    if longest > max_px:
        zoom *= max_px / longest
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    if image_format == "png":
        data = pixmap.tobytes("png")
    else:
        data = pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)
    return data, pixmap.width, pixmap.height


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


# =========================================================================================
# One document
# =========================================================================================


def prepare_sync(pdf_bytes: bytes, settings: Settings) -> PreparedDoc:
    """Classify, skip, extract text and render images (blocking: use `prepare`)."""
    start = time.perf_counter()
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    mime = f"image/{settings.image_format}"
    with _PDF_LOCK:
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise ValueError(f"not a readable PDF: {exc}") from exc
        with doc:
            texts = [page.get_text() for page in doc]
            kinds = [classify(text, settings.native_min_chars) for text in texts]
            skips: list[str | None] = [
                fieldless_reason(text)
                if settings.skip_fieldless_pages and kind == "native" and index > 0
                else None
                for index, (text, kind) in enumerate(zip(texts, kinds, strict=True))
            ]
            timings = {"classify": _ms(start)}

            text_start = time.perf_counter()
            with_text = [
                i
                for i, kind in enumerate(kinds)
                if kind == "native" and skips[i] is None and settings.native_mode != "image"
            ]
            if settings.text_engine == "pymupdf4llm" and with_text:
                page_texts = _markdown_pages(doc, with_text)
            else:
                page_texts = {i: texts[i].strip() for i in with_text}
            timings["text"] = _ms(text_start)

            render_start = time.perf_counter()
            pages = []
            for i, page in enumerate(doc):
                image = None
                wants_image = kinds[i] == "scanned" or settings.native_mode != "text"
                if skips[i] is None and wants_image:
                    image = render_page(
                        page,
                        settings.image_dpi,
                        settings.image_format,
                        settings.image_jpeg_quality,
                        settings.max_image_px,
                    )
                pages.append(
                    PageContent(
                        page_no=i + 1,
                        kind=kinds[i],
                        text_md=page_texts.get(i),
                        image_b64=base64.b64encode(image[0]).decode() if image else None,
                        image_mime=mime if image else None,
                        width=image[1] if image else None,
                        height=image[2] if image else None,
                        skipped=skips[i] is not None,
                        skip_reason=skips[i],
                    )
                )
            timings["render"] = _ms(render_start)
    timings["total"] = _ms(start)
    return PreparedDoc(doc_sha256=sha256, page_count=len(pages), pages=pages, timings_ms=timings)


async def prepare(pdf_bytes: bytes, settings: Settings) -> PreparedDoc:
    """Pre-process a PDF off the event loop (PyMuPDF work runs in a worker thread)."""
    return await asyncio.to_thread(prepare_sync, pdf_bytes, settings)
