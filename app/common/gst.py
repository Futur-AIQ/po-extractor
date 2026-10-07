"""Indian GST identifiers: state codes, GSTIN checksum and PAN format (PRD §9.4 rule 2).

Shared by the extraction pipeline (hints, validation) and the synthetic data generator, so
both use exactly the same rules.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """An Indian state or union territory with its official 2-digit GST state code."""

    code: str
    name: str


# Official GST state codes (GSTN). Legacy codes 25 (Daman & Diu, merged into 26 in 2020) and
# 28 (undivided Andhra Pradesh, now 37) are accepted by the validator but not generated.
STATES: dict[str, State] = {
    s.code: s
    for s in [
        State("01", "Jammu and Kashmir"),
        State("02", "Himachal Pradesh"),
        State("03", "Punjab"),
        State("04", "Chandigarh"),
        State("05", "Uttarakhand"),
        State("06", "Haryana"),
        State("07", "Delhi"),
        State("08", "Rajasthan"),
        State("09", "Uttar Pradesh"),
        State("10", "Bihar"),
        State("11", "Sikkim"),
        State("12", "Arunachal Pradesh"),
        State("13", "Nagaland"),
        State("14", "Manipur"),
        State("15", "Mizoram"),
        State("16", "Tripura"),
        State("17", "Meghalaya"),
        State("18", "Assam"),
        State("19", "West Bengal"),
        State("20", "Jharkhand"),
        State("21", "Odisha"),
        State("22", "Chhattisgarh"),
        State("23", "Madhya Pradesh"),
        State("24", "Gujarat"),
        State("26", "Dadra and Nagar Haveli and Daman and Diu"),
        State("27", "Maharashtra"),
        State("29", "Karnataka"),
        State("30", "Goa"),
        State("31", "Lakshadweep"),
        State("32", "Kerala"),
        State("33", "Tamil Nadu"),
        State("34", "Puducherry"),
        State("35", "Andaman and Nicobar Islands"),
        State("36", "Telangana"),
        State("37", "Andhra Pradesh"),
        State("38", "Ladakh"),
    ]
}
LEGACY_STATE_CODES = {"25", "28"}
SPECIAL_STATE_CODES = {"97", "99"}  # 97 Other Territory, 99 Centre Jurisdiction
VALID_GSTIN_STATE_CODES = set(STATES) | LEGACY_STATE_CODES | SPECIAL_STATE_CODES

_ALNUM36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_FORMAT = re.compile(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]")
PAN_FORMAT = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")

# PAN 4th character = holder type: A association of persons, B body of individuals,
# C company, F firm/LLP, G government, H HUF, J artificial juridical person,
# L local authority, P individual, T trust.
PAN_HOLDER_TYPES = frozenset("ABCFGHJLPT")


def gstin_check_char(first14: str) -> str:
    """Official GSTIN check character (mod 36) for the first 14 characters."""
    if len(first14) != 14 or any(c not in _ALNUM36 for c in first14):
        raise ValueError(f"expected 14 characters from 0-9A-Z, got {first14!r}")
    total = 0
    for index, char in enumerate(first14):
        product = _ALNUM36.index(char) * (1 if index % 2 == 0 else 2)
        total += product // 36 + product % 36
    return _ALNUM36[(36 - total % 36) % 36]


def is_valid_gstin(gstin: str) -> bool:
    """True if `gstin` has the GSTIN format, a known state code and a correct check character."""
    if not isinstance(gstin, str) or not GSTIN_FORMAT.fullmatch(gstin):
        return False
    if gstin[:2] not in VALID_GSTIN_STATE_CODES:
        return False
    return gstin_check_char(gstin[:14]) == gstin[14]


def pan_from_gstin(gstin: str) -> str:
    """The PAN embedded in a GSTIN (characters 3-12). Raises ValueError if the GSTIN is invalid."""
    if not is_valid_gstin(gstin):
        raise ValueError(f"invalid GSTIN {gstin!r}")
    return gstin[2:12]


def is_valid_pan(pan: str) -> bool:
    """True if `pan` has the PAN format with a known holder type (4th character).

    The last character's algorithm is not public, so it cannot be checked.
    """
    return isinstance(pan, str) and bool(PAN_FORMAT.fullmatch(pan)) and pan[3] in PAN_HOLDER_TYPES
