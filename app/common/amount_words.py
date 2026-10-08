"""Read an amount written in words, Indian system, back into a number (PRD §9.4 rule 7).

    parse_amount_in_words("Rupees Two Lakh Forty Five Thousand Three Hundred Twelve and
                           Fifty Paise Only")  ->  Decimal("245312.50")

Understands Crore / Lakh (Lac) / Thousand / Hundred, plural forms, "and", hyphens and
commas, a currency word in front (Rupees, Rs, INR, Indian Rupees) and "Only" at the end.
"""

import re
from decimal import Decimal

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_SCALES = {
    "crore": 10_000_000, "crores": 10_000_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "thousand": 1_000, "thousands": 1_000,
}  # fmt: skip
_IGNORED = {"rupees", "rupee", "rs", "inr", "indian", "only", "of"}


def _words_to_int(words: list[str]) -> int:
    """'two lakh forty five thousand three hundred twelve' -> 245312 (Indian scales)."""
    if not words:
        raise ValueError("no number words")
    total, group = 0, 0  # group: the number below the next scale word
    for word in words:
        if word in _UNITS:
            group += _UNITS[word]
        elif word == "hundred":
            group = (group or 1) * 100
        elif word in _SCALES:
            if word.startswith("crore"):  # "one lakh crore": everything so far is crores
                total, group = (total + group) * _SCALES[word], 0
            else:
                total, group = total + (group or 1) * _SCALES[word], 0
        else:
            raise ValueError(f"not a number word: {word!r}")
    return total + group


def parse_amount_in_words(text: str) -> Decimal:
    """The amount in `text` as a Decimal. Raises ValueError if it cannot be read.

    Paise may follow the rupees as "... and Fifty Paise" or "... and Paise Fifty".
    """
    words = [w for w in re.sub(r"[^a-z]+", " ", text.lower()).split() if w not in _IGNORED]
    paise_at = [i for i, word in enumerate(words) if word in ("paise", "paisa")]
    if len(paise_at) > 1:
        raise ValueError(f"paise mentioned twice in {text!r}")
    if not paise_at:
        rupee_words, paise_words = words, []
    elif paise_at[0] < len(words) - 1:  # "... and Paise Fifty"
        rupee_words, paise_words = words[: paise_at[0]], words[paise_at[0] + 1 :]
    else:  # "... and Fifty Paise": the paise number follows the last "and"
        before = words[: paise_at[0]]
        cut = len(before) - before[::-1].index("and") - 1 if "and" in before else -1
        rupee_words, paise_words = before[: max(cut, 0)], before[cut + 1 :]
    rupee_words = [w for w in rupee_words if w != "and"]  # "One Hundred and Five"
    paise_words = [w for w in paise_words if w != "and"]
    if not rupee_words and not paise_words:
        raise ValueError(f"no amount in {text!r}")
    rupees = _words_to_int(rupee_words) if rupee_words else 0
    paise = _words_to_int(paise_words) if paise_words else 0
    if paise >= 100:
        raise ValueError(f"paise must be below 100 in {text!r}")
    return Decimal(rupees) + Decimal(paise) / 100
