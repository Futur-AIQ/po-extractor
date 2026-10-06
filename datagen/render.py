"""HTML -> A4 PDF with headless Chromium, plus PDF inspection helpers (PyMuPDF).

    async with PdfRenderer(max_pages=4) as renderer:
        pdf_bytes = await renderer.render(html)

One browser and one context serve every render; up to `max_pages` tabs render concurrently.
Page size and margins come from the template's CSS (@page), so printing matches the layout.
"""

import asyncio
from types import TracebackType

import pymupdf
from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright

from datagen.templates import format_money
from schema.po_schema import PurchaseOrder


class PdfRenderer:
    """Async Chromium PDF renderer; use as an async context manager."""

    def __init__(self, max_pages: int = 4) -> None:
        self._limit = asyncio.Semaphore(max_pages)
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> "PdfRenderer":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        self._context = await self._browser.new_context()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    async def render(self, html: str) -> bytes:
        """Render a complete HTML document to PDF bytes (A4, backgrounds printed)."""
        if self._context is None:
            raise RuntimeError("PdfRenderer must be used inside 'async with'")
        async with self._limit:
            page = await self._context.new_page()
            try:
                await page.set_content(html, wait_until="load")
                return await page.pdf(
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=True,  # @page size and margins from the template
                )
            finally:
                await page.close()


# =========================================================================================
# PDF inspection (synchronous PyMuPDF; call via asyncio.to_thread from async code)
# =========================================================================================


def page_texts(pdf: bytes) -> list[str]:
    """Extracted text of every page."""
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [page.get_text() for page in doc]


def lines_per_page(po: PurchaseOrder, texts: list[str]) -> list[list[int]]:
    """Which line_no values are printed on each page.

    A line belongs to the page where its taxable value and line total are both printed (the
    end of its row). Pages are assigned in document order, so a value repeated later (e.g. in
    a summary) cannot pull a line onto the wrong page. Unmatched lines are left out; callers
    compare the total against len(po.line_items).
    """
    mapping: list[list[int]] = [[] for _ in texts]
    page = 0
    for item in po.line_items:
        keys = (format_money(item.taxable_value), format_money(item.line_total))
        for candidate in range(page, len(texts)):
            if all(key in texts[candidate] for key in keys):
                mapping[candidate].append(item.line_no)
                page = candidate
                break
    return mapping
