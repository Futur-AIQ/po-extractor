"""Tests for realistic Indian business data (datagen/india.py, Step 1.2)."""

import random
import re
from collections import Counter
from decimal import Decimal

import pytest

from datagen import india
from datagen.india import (
    CATALOGUE,
    CITIES,
    GST_SLABS,
    INDUSTRIES,
    STATES,
    UOMS,
    amount_in_words_inr,
    generate_address,
    generate_company,
    generate_contact,
    generate_gstin,
    generate_item_code,
    generate_pan,
    generate_unit_price,
    gstin_check_char,
    is_valid_gstin,
    items_for,
    make_faker,
    pan_from_gstin,
    pick_gst_rate,
    pick_industrial_state,
)

# Example GSTIN used in official GST documentation; its check character is correct.
KNOWN_VALID_GSTIN = "27AAPFU0939F1ZV"

# First digit of the PIN code = postal zone of the state.
POSTAL_ZONE = {"07": "1", "06": "1", "09": "2", "08": "3", "24": "3", "27": "4", "29": "5",
               "36": "5", "33": "6", "19": "7"}  # fmt: skip


# --- States and cities -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "name"),
    [("24", "Gujarat"), ("27", "Maharashtra"), ("29", "Karnataka"), ("33", "Tamil Nadu"),
     ("07", "Delhi"), ("36", "Telangana"), ("37", "Andhra Pradesh"), ("38", "Ladakh")],
)  # fmt: skip
def test_official_state_codes(code: str, name: str) -> None:
    assert STATES[code].name == name


def test_all_states_and_uts_present() -> None:
    assert len(STATES) == 36  # 28 states + 8 UTs
    assert all(re.fullmatch(r"\d{2}", code) for code in STATES)
    assert "25" not in STATES and "28" not in STATES  # legacy codes are not generated


def test_industrial_states_have_cities_with_consistent_pins() -> None:
    assert len(CITIES) == 10
    for code, cities in CITIES.items():
        assert code in STATES
        assert 4 <= len(cities) <= 6, STATES[code].name
        for city in cities:
            assert re.fullmatch(r"\d{3}", city.pin_prefix)
            assert city.pin_prefix[0] == POSTAL_ZONE[code], f"{city.name} PIN zone"
            assert city.estates


def test_address_is_consistent_with_state() -> None:
    rng = random.Random(1)
    for code in CITIES:
        state = STATES[code]
        address = generate_address(rng, state)
        city = next(c for c in CITIES[code] if c.name == address.city)
        assert re.fullmatch(r"\d{6}", address.pin)
        assert address.pin.startswith(city.pin_prefix)
        assert address.lines[-1] == state.name
        assert f"{city.name} - {address.pin}" in address.one_line()


def test_address_for_state_without_city_data_raises() -> None:
    with pytest.raises(ValueError, match="no city data"):
        generate_address(random.Random(1), STATES["11"])


def test_pick_industrial_state_only_returns_states_with_cities() -> None:
    rng = random.Random(3)
    assert {pick_industrial_state(rng).code for _ in range(500)} == set(CITIES)


# --- GSTIN and PAN -----------------------------------------------------------------------


def test_known_valid_gstin_validates() -> None:
    assert is_valid_gstin(KNOWN_VALID_GSTIN)
    assert gstin_check_char(KNOWN_VALID_GSTIN[:14]) == "V"
    assert pan_from_gstin(KNOWN_VALID_GSTIN) == "AAPFU0939F"


def test_any_single_character_change_fails() -> None:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for position, original in enumerate(KNOWN_VALID_GSTIN):
        for replacement in alphabet:
            if replacement == original:
                continue
            mutated = KNOWN_VALID_GSTIN[:position] + replacement + KNOWN_VALID_GSTIN[position + 1 :]
            assert not is_valid_gstin(mutated), mutated


@pytest.mark.parametrize(
    "bad",
    ["", "27AAPFU0939F1Z", "27AAPFU0939F1ZVX", "27aapfu0939f1zv", "00AAPFU0939F1ZV", 27, None],
)
def test_malformed_gstins_are_invalid(bad: object) -> None:
    assert not is_valid_gstin(bad)  # type: ignore[arg-type]


def test_pan_from_invalid_gstin_raises() -> None:
    with pytest.raises(ValueError, match="invalid GSTIN"):
        pan_from_gstin("27AAPFU0939F1ZW")


def test_generated_gstins_validate_for_every_state() -> None:
    rng = random.Random(7)
    for code in STATES:
        for _ in range(20):
            pan = generate_pan(rng)
            gstin = generate_gstin(rng, code, pan)
            assert is_valid_gstin(gstin), gstin
            assert gstin[:2] == code
            assert pan_from_gstin(gstin) == pan
            assert gstin[13] == "Z"


def test_company_pan_format() -> None:
    rng = random.Random(11)
    pan = generate_pan(rng, "C", "Patel Engineering Works Pvt. Ltd.")
    assert re.fullmatch(r"[A-Z]{3}CP[0-9]{4}[A-Z]", pan)


def test_generate_gstin_rejects_bad_input() -> None:
    rng = random.Random(1)
    with pytest.raises(ValueError, match="unknown GST state code"):
        generate_gstin(rng, "99", "AAPFU0939F")
    with pytest.raises(ValueError, match="invalid PAN"):
        generate_gstin(rng, "27", "AAPF0939F")


# --- Companies and contacts ----------------------------------------------------------------


def test_company_legal_form_matches_pan_holder_type() -> None:
    rng = random.Random(5)
    for _ in range(300):
        company = generate_company(rng, rng.choice(INDUSTRIES))
        if company.name.endswith(("Pvt. Ltd.", "Private Limited", "Ltd.", "Limited")):
            assert company.pan_holder_type == "C", company.name
        elif company.name.endswith(("LLP", "& Co.")):
            assert company.pan_holder_type == "F", company.name
        else:
            assert company.pan_holder_type == "P", company.name


def test_company_name_fits_industry() -> None:
    rng = random.Random(9)
    names = [generate_company(rng, "chemicals").name for _ in range(50)]
    sector_words = ("Chem", "Polymers", "Petrochem", "& Co.")
    assert all(any(word in name for word in sector_words) for name in names)


def test_unknown_industry_raises() -> None:
    with pytest.raises(ValueError, match="unknown industry"):
        generate_company(random.Random(1), "aerospace")


def test_contact_uses_company_domain() -> None:
    rng = random.Random(2)
    company = generate_company(rng, "electrical")
    contact = generate_contact(make_faker(rng), company.domain)
    assert contact.email.endswith("@" + company.domain)
    assert re.fullmatch(r"[a-z0-9.]+@[a-z0-9.]+", contact.email)
    assert contact.name and contact.phone


# --- Determinism -----------------------------------------------------------------------------


def _sample(seed: int) -> list[object]:
    """One of everything, generated from a single seed."""
    rng = random.Random(seed)
    fake = make_faker(rng)
    out: list[object] = []
    for industry in INDUSTRIES:
        state = pick_industrial_state(rng)
        company = generate_company(rng, industry)
        pan = generate_pan(rng, company.pan_holder_type, company.name)
        item = rng.choice(items_for(industry))
        out += [
            company,
            generate_gstin(rng, state.code, pan),
            generate_address(rng, state),
            generate_contact(fake, company.domain),
            generate_item_code(rng, item),
            generate_unit_price(rng, item),
            pick_gst_rate(rng),
        ]
    return out


def test_same_seed_same_data() -> None:
    assert _sample(42) == _sample(42)
    assert _sample(42) != _sample(43)


def test_global_random_is_never_used(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("global random module used")

    for name in ("random", "randint", "choice", "choices", "uniform", "shuffle", "sample"):
        monkeypatch.setattr(random, name, forbidden)
    assert _sample(1)


# --- Catalogue and GST ---------------------------------------------------------------------


def test_catalogue_size_and_industries() -> None:
    assert 110 <= len(CATALOGUE) <= 140
    counts = Counter(item.industry for item in CATALOGUE)
    assert set(counts) == set(INDUSTRIES)
    assert all(n >= 15 for n in counts.values()), counts


def test_catalogue_items_are_well_formed() -> None:
    for item in CATALOGUE:
        assert item.uom in UOMS, item.description
        assert re.fullmatch(r"\d{4}|\d{6}|\d{8}", item.hsn_sac), item.description
        assert item.is_service == (item.industry == "services"), item.description
        assert Decimal(0) < item.price_min < item.price_max, item.description
        assert item.gst_rate in GST_SLABS, f"{item.description}: {item.gst_rate}% not a slab"
        assert re.fullmatch(r"[A-Z]{3}", item.code_prefix)


def test_catalogue_has_long_descriptions() -> None:
    # Long descriptions wrap to several lines in the rendered PDF (a difficulty knob).
    long_ones = [item for item in CATALOGUE if len(item.description) > 90]
    assert len(long_ones) >= 10
    assert {item.industry for item in long_ones} == set(INDUSTRIES)


def test_most_items_are_18_percent() -> None:
    rates = Counter(item.gst_rate for item in CATALOGUE)
    assert rates.most_common(1)[0][0] == Decimal(18)
    assert rates[Decimal(18)] / len(CATALOGUE) > 0.8


def test_gst_slabs_default_structure() -> None:
    assert set(GST_SLABS) == {Decimal(0), Decimal(5), Decimal(18), Decimal(40)}
    assert max(GST_SLABS, key=GST_SLABS.__getitem__) == Decimal(18)


def test_pick_gst_rate_follows_weights() -> None:
    rng = random.Random(4)
    counts = Counter(pick_gst_rate(rng) for _ in range(5000))
    assert set(counts) <= set(GST_SLABS)
    assert counts[Decimal(18)] > counts[Decimal(5)] > counts[Decimal(40)]


def test_unit_price_in_range_with_two_decimals() -> None:
    rng = random.Random(6)
    for item in CATALOGUE:
        price = generate_unit_price(rng, item)
        assert isinstance(price, Decimal)
        assert price.as_tuple().exponent == -2
        assert item.price_min - 1 <= price <= item.price_max


def test_item_code_styles() -> None:
    rng = random.Random(8)
    item = items_for("industrial_equipment")[0]
    codes = {generate_item_code(rng, item) for _ in range(200)}
    assert any(re.fullmatch(r"BRG-\d{3,4}", c) for c in codes)
    assert any(re.fullmatch(r"BRG/\d{4}/[ABC]", c) for c in codes)
    assert any(re.fullmatch(r"1\d{7}", c) for c in codes)


# --- Amount in words -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amount", "words"),
    [
        ("245312.50",
         "Rupees Two Lakh Forty Five Thousand Three Hundred Twelve and Fifty Paise Only"),
        ("100000", "Rupees One Lakh Only"),
        ("10000000", "Rupees One Crore Only"),
        ("12345678.09",
         "Rupees One Crore Twenty Three Lakh Forty Five Thousand Six Hundred Seventy Eight "
         "and Nine Paise Only"),
        ("1234567890.05",
         "Rupees One Hundred Twenty Three Crore Forty Five Lakh Sixty Seven Thousand "
         "Eight Hundred Ninety and Five Paise Only"),
        ("1000000000000", "Rupees One Lakh Crore Only"),
        ("1001", "Rupees One Thousand One Only"),
        ("0.75", "Rupees Zero and Seventy Five Paise Only"),
        ("33024.00", "Rupees Thirty Three Thousand Twenty Four Only"),
    ],
)  # fmt: skip
def test_amount_in_words_indian_system(amount: str, words: str) -> None:
    assert amount_in_words_inr(Decimal(amount)) == words


def test_amount_in_words_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="negative"):
        amount_in_words_inr(Decimal("-1"))
    with pytest.raises(ValueError, match="2 decimal places"):
        amount_in_words_inr(Decimal("10.005"))
    with pytest.raises(TypeError, match="Decimal"):
        amount_in_words_inr(10.5)  # type: ignore[arg-type]


# --- Text pools ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pool",
    ["PAYMENT_TERMS", "DELIVERY_TERMS", "FREIGHT_TERMS", "TRANSPORT_MODES", "WARRANTY_TERMS",
     "PACKING_INSTRUCTIONS"],
)  # fmt: skip
def test_text_pools(pool: str) -> None:
    values = getattr(india, pool)
    assert len(values) >= 5
    assert len(set(values)) == len(values)
    assert all(v.strip() == v and v for v in values)
