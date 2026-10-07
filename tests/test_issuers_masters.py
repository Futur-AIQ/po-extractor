"""Tests for issuer/counterparty pools and mock ERP masters (Step 1.4b)."""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from app.common.gst import is_valid_gstin, pan_from_gstin
from datagen.consistency import check_po
from datagen.generate import GeneratedPO, Knobs, generate_many
from datagen.india import CATALOGUE, INDUSTRIAL_STATE_WEIGHTS
from datagen.issuers import (
    DATE_STYLES,
    LAYOUT_IDS,
    PO_NUMBER_STYLES,
    counterparties,
    frequent_issuers,
    make_one_off_issuer,
    serial_range,
    vendor_code,
)
from datagen.masters import HISTORY_MAX, HISTORY_MIN, build_masters, write_masters
from datagen.templates import LAYOUTS, format_date
from schema.dates import parse_po_date

PO_NUMBER_PATTERNS = {
    "fy_slash": r"PO/\d{4}-\d{2}/\d{5}",
    "sap": r"45000\d{5}",
    "dept": r"[A-Z]{3}-PUR-\d{2}-\d{4}",
    "unit_fy": r"[A-Z]{3}/PO/\d{1,4}/\d{2}-\d{2}",
    "yearmonth": r"PO-\d{6}-\d{4}",
}
FINGERPRINT_CODE = """
import hashlib
from datagen.issuers import frequent_issuers, counterparties
parts = [f"{i.id}|{i.company.name}|{'|'.join(u.gstin for u in i.units)}|{i.layout}|"
         f"{i.po_number_style}|{i.date_style}|{i.item_code_style}|{i.item_code_base}"
         for i in frequent_issuers()]
parts += [f"{c.id}|{c.company.name}|{c.unit.gstin}" for c in counterparties()]
print(hashlib.sha256("\\n".join(parts).encode()).hexdigest())
"""


@pytest.fixture(scope="module")
def dataset() -> list[GeneratedPO]:
    return generate_many(500, seed=2026)


@pytest.fixture(scope="module")
def small() -> list[GeneratedPO]:
    return generate_many(60, seed=42)


# --- Pools ---------------------------------------------------------------------------------------


def _fingerprint_in_fresh_process(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    result = subprocess.run(
        [sys.executable, "-c", FINGERPRINT_CODE], capture_output=True, text=True, env=env,
        cwd=Path(__file__).parents[1], check=True,
    )  # fmt: skip
    return result.stdout.strip()


def test_pools_identical_across_runs() -> None:
    first = _fingerprint_in_fresh_process("1")
    second = _fingerprint_in_fresh_process("2")
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_pools_independent_of_dataset_seed() -> None:
    before = frequent_issuers.__wrapped__(), counterparties.__wrapped__()
    generate_many(20, seed=1)
    generate_many(20, seed=999)
    assert (frequent_issuers(), counterparties()) == before


def test_frequent_issuer_pool_shape() -> None:
    issuers = frequent_issuers()
    assert len(issuers) == 30
    assert len({i.id for i in issuers}) == 30
    assert Counter(i.layout for i in issuers) == dict.fromkeys(LAYOUT_IDS, 5)
    assert tuple(LAYOUTS) == LAYOUT_IDS
    for issuer in issuers:
        assert issuer.frequent
        assert issuer.po_number_style in PO_NUMBER_STYLES
        assert issuer.date_style in DATE_STYLES
        assert len(issuer.units) == 3
        for unit in issuer.units:  # one PAN, one GSTIN per state registration
            assert is_valid_gstin(unit.gstin)
            assert pan_from_gstin(unit.gstin) == issuer.pan
            assert unit.gstin[:2] == unit.state.code
            assert unit.address.state == unit.state


def test_date_styles_are_renderable_and_parseable() -> None:
    from datetime import date

    for style in DATE_STYLES:
        assert parse_po_date(format_date(date(2026, 3, 14), style)) == date(2026, 3, 14)


def test_counterparty_pool_covers_every_industrial_state() -> None:
    pool = counterparties()
    assert len(pool) == 40
    assert Counter(c.unit.state.code for c in pool) == dict.fromkeys(INDUSTRIAL_STATE_WEIGHTS, 4)
    assert all(is_valid_gstin(c.unit.gstin) for c in pool)


def test_issuer_item_codes_are_unique_and_stable() -> None:
    for issuer in frequent_issuers():
        codes = [issuer.item_code(item) for item in CATALOGUE]
        assert len(set(codes)) == len(codes), issuer.id
        assert codes == [issuer.item_code(item) for item in CATALOGUE]


def test_one_off_issuer() -> None:
    issuer = make_one_off_issuer(random.Random(5))
    assert not issuer.frequent and issuer.id.startswith("ONE-")
    assert issuer.layout in LAYOUT_IDS
    assert make_one_off_issuer(random.Random(5)) == issuer
    assert make_one_off_issuer(random.Random(6)) != issuer


def test_vendor_code_is_fixed_per_issuer_and_vendor() -> None:
    issuer = frequent_issuers()[0]
    vendor = counterparties()[0]
    assert vendor_code(issuer, vendor) == vendor_code(issuer, vendor)
    assert re.fullmatch(r"V\d{5}|SUP-\d{4}", vendor_code(issuer, vendor))


@pytest.mark.parametrize("style", list(PO_NUMBER_STYLES))
def test_historical_and_new_serials_never_overlap(style: str) -> None:
    old_low, old_high = serial_range(style, historical=True)
    new_low, new_high = serial_range(style, historical=False)
    assert old_low >= 1 and old_high < new_low <= new_high < 10 ** PO_NUMBER_STYLES[style]


# --- Distribution ------------------------------------------------------------------------------


def test_frequent_issuer_share_about_70_percent(dataset: list[GeneratedPO]) -> None:
    share = sum(g.meta.issuer_frequent for g in dataset) / len(dataset)
    assert abs(share - 0.70) < 0.05


def test_line_count_distribution(dataset: list[GeneratedPO]) -> None:
    counts = [len(g.po.line_items) for g in dataset]
    assert min(counts) >= 20 and max(counts) <= 80
    assert sum(30 <= c <= 50 for c in counts) / len(counts) > 0.7  # most POs 30-50
    assert any(c < 30 for c in counts) and any(c > 50 for c in counts)


def test_pos_follow_their_issuer(dataset: list[GeneratedPO]) -> None:
    pool = {i.id: i for i in frequent_issuers()}
    vendors = {c.unit.gstin for c in counterparties()}
    for g in dataset:
        issuer, po = g.issuer, g.po
        if g.meta.issuer_frequent:
            assert pool[g.meta.issuer_id] == issuer
        assert g.meta.layout == issuer.layout
        assert g.meta.issuer_date_style == issuer.date_style
        assert po.buyer_gstin == issuer.main.gstin and po.buyer_name == issuer.company.name
        assert {po.bill_to_gstin, po.ship_to_gstin} - {None} <= {u.gstin for u in issuer.units}
        assert po.vendor_gstin in vendors
        assert re.fullmatch(PO_NUMBER_PATTERNS[issuer.po_number_style], po.po_number)
        for item in po.line_items:
            assert item.item_code is None or item.item_code in {
                issuer.item_code(c) for c in CATALOGUE
            }


def test_po_numbers_unique_per_issuer(dataset: list[GeneratedPO]) -> None:
    keys = [(g.meta.issuer_id, g.po.po_number) for g in dataset]
    assert len(keys) == len(set(keys))


def test_consistency_still_holds(dataset: list[GeneratedPO]) -> None:
    assert [v for g in dataset for v in check_po(g.po)] == []


# --- Duplicates and masters -----------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 2, 5])
def test_exactly_duplicate_po_count_expected_duplicates(count: int) -> None:
    pos = generate_many(60, seed=7, knobs=Knobs(duplicate_po_count=count))
    marked = [g for g in pos if g.meta.expected_duplicate]
    assert len(marked) == count
    assert all(g.meta.issuer_frequent for g in marked)


def test_masters_cover_every_party_and_item(small: list[GeneratedPO]) -> None:
    masters = build_masters(small, frequent_issuers(), counterparties(), CATALOGUE)
    items = {(r["issuer_gstin"], r["item_code"]) for r in masters.items}
    for g in small:
        po = g.po
        parties = [
            (po.buyer_gstin, po.buyer_name), (po.vendor_gstin, po.vendor_name),
            (po.bill_to_gstin, po.bill_to_name), (po.ship_to_gstin, po.ship_to_name),
        ]  # fmt: skip
        for gstin, name in parties:
            if gstin is not None:
                assert masters.parties[gstin]["name"] == name
                assert masters.parties[gstin]["state_code"] == gstin[:2]
        for item in po.line_items:
            if item.item_code is not None:
                assert (po.buyer_gstin, item.item_code) in items


def test_masters_include_every_catalogue_item_per_issuer(small: list[GeneratedPO]) -> None:
    masters = build_masters(small, frequent_issuers(), counterparties(), CATALOGUE)
    per_issuer = Counter(r["issuer_id"] for r in masters.items)
    assert all(per_issuer[i.id] == len(CATALOGUE) for i in frequent_issuers())


def test_only_deliberate_duplicates_are_already_processed(small: list[GeneratedPO]) -> None:
    masters = build_masters(small, frequent_issuers(), counterparties(), CATALOGUE)
    processed = {
        (gstin, number)
        for gstin, entry in masters.processed_po_numbers.items()
        for number in entry["po_numbers"]
    }
    hits = [g for g in small if (g.po.buyer_gstin, g.po.po_number) in processed]
    assert hits == [g for g in small if g.meta.expected_duplicate]
    assert len(hits) == Knobs().duplicate_po_count


def test_historical_numbers_per_frequent_issuer(small: list[GeneratedPO]) -> None:
    masters = build_masters(small, frequent_issuers(), counterparties(), CATALOGUE)
    for issuer in frequent_issuers():
        numbers = masters.processed_po_numbers[issuer.main.gstin]["po_numbers"]
        assert HISTORY_MIN <= len(numbers) <= HISTORY_MAX + Knobs().duplicate_po_count
        assert len(numbers) == len(set(numbers))
        assert all(re.fullmatch(PO_NUMBER_PATTERNS[issuer.po_number_style], n) for n in numbers)


def test_write_masters(small: list[GeneratedPO], tmp_path: Path) -> None:
    masters = build_masters(small, frequent_issuers(), counterparties(), CATALOGUE)
    paths = write_masters(masters, tmp_path)
    assert [p.name for p in paths] == ["parties.json", "items.json", "processed_po_numbers.json"]
    parties = json.loads(paths[0].read_text())
    assert {p["gstin"] for p in parties} == set(masters.parties)
    assert len(json.loads(paths[1].read_text())) == len(masters.items)
    first = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
    write_masters(build_masters(small, frequent_issuers(), counterparties(), CATALOGUE), tmp_path)
    assert hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest() == first
