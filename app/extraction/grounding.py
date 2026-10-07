"""Grounding: is an extracted identifier printed in the PO's text layer? (PRD §9.4 rule 9)

The index holds the native text layers as one normalised corpus: uppercase, with spaces and
separators removed, so "PO/2025-26/85782" matches a PO number that wraps after "PO/2025-".

is_grounded(value):
    "yes"           found in the corpus
    "no"            not found, and every page sent to the LLM has a text layer in the index:
                    the value is not printed, so the model made it up or misread it
    "unverifiable"  not found, but some page has no text layer (scanned, or a native page in
                    native_mode "image"): the value may be printed there. Never a failure.

Short values (fewer than 8 characters, such as a 4-digit HSN) must also start and end at a
token boundary of the original text; anywhere in a long digit run, "8483" would match by
chance. Pages skipped as field-less are not indexed and do not make a PO unverifiable: they
carry no PO fields and are never sent to the LLM.
"""

import re
from collections.abc import Iterable
from typing import Literal

from app.extraction.preprocess import PreparedDoc

Grounded = Literal["yes", "no", "unverifiable"]

_TOKEN = re.compile(r"[A-Z0-9]+")
MIN_FREE_MATCH_LEN = 8  # shorter values must match whole tokens


def normalise(value: str) -> str:
    """Uppercase, spaces and separators removed: "po/2025-26 / 01" -> "PO20252601"."""
    return "".join(_TOKEN.findall(value.upper()))


class GroundingIndex:
    """Normalised text of a PO's native pages, for checking extracted identifiers."""

    def __init__(self, texts: Iterable[str], complete: bool) -> None:
        """`texts`: one text layer per page; `complete`: True if every page has one."""
        self.complete = complete
        parts: list[str] = []
        self._boundaries = {0}  # offsets in the corpus where a token starts or ends
        length = 0
        for text in texts:
            for token in _TOKEN.findall(text.upper()):
                parts.append(token)
                length += len(token)
                self._boundaries.add(length)
        self._corpus = "".join(parts)

    @classmethod
    def from_doc(cls, doc: PreparedDoc) -> "GroundingIndex":
        """Index the text layers of the pages sent to the LLM."""
        pages = doc.kept_pages
        return cls(
            (page.text_md for page in pages if page.text_md),
            complete=all(page.text_md for page in pages),
        )

    def contains(self, value: str) -> bool:
        """True if the normalised `value` occurs in the corpus (see module docstring)."""
        needle = normalise(value)
        if not needle:
            return False
        if len(needle) >= MIN_FREE_MATCH_LEN:
            return needle in self._corpus
        start = self._corpus.find(needle)
        while start != -1:
            if start in self._boundaries and start + len(needle) in self._boundaries:
                return True
            start = self._corpus.find(needle, start + 1)
        return False

    def is_grounded(self, value: str) -> Grounded:
        """Grounding of an extracted identifier: "yes", "no" or "unverifiable" (see module doc).

        A value without letters or digits cannot be checked and is "unverifiable".
        """
        if self.contains(value):
            return "yes"
        if self.complete and normalise(value):
            return "no"
        return "unverifiable"
