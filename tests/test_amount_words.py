"""Tests for reading amounts in words (app/common/amount_words.py, Step 3.6)."""

import random
from decimal import Decimal

import pytest

from app.common.amount_words import parse_amount_in_words
from datagen.india import amount_in_words_inr


def test_round_trip_with_the_generator() -> None:
    rng = random.Random(7)
    amounts = [Decimal(rng.randint(1, 10**10)) / 100 for _ in range(200)]
    amounts += [Decimal(n) for n in (1, 10, 100, 101, 1000, 100000, 10**7, 10**9, 12_34_56_789)]
    for amount in amounts:
        assert parse_amount_in_words(amount_in_words_inr(amount)) == amount, amount


@pytest.mark.parametrize(
    ("text", "amount"),
    [
        ("Rupees Sixty Two Lakh Thirty One Thousand One Hundred Twenty Three Only", "6231123"),
        ("Rs. One Crore Two Lakhs Only", "10200000"),
        ("INR one hundred and five only", "105"),
        ("Rupees Sixty-Two Thousand, Five Hundred Only", "62500"),
        ("Indian Rupees Ten Lacs Only", "1000000"),
        ("Rupees Twenty and Five Paise Only", "20.05"),
        ("Rupees Ten and Paise Ninety Nine Only", "10.99"),
        ("Rupees Fifty Paise Only", "0.50"),
        ("RUPEES ONE LAKH CRORE ONLY", "1000000000000"),
    ],
)
def test_variants(text: str, amount: str) -> None:
    assert parse_amount_in_words(text) == Decimal(amount)


@pytest.mark.parametrize(
    "text", ["", "Rupees Only", "Rupees Twelve Apples Only", "Rupees Ten and Hundred Fifty Paise"]
)
def test_unreadable_text_raises(text: str) -> None:
    with pytest.raises(ValueError):
        parse_amount_in_words(text)
