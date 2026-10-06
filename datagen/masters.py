"""Mock ERP master data for validation rule 10 (PRD §9.4).

    masters = build_masters(pos, frequent_issuers(), counterparties(), CATALOGUE)
    write_masters(masters, out_dir)   # parties.json, items.json, processed_po_numbers.json

- parties: every registration (GSTIN) of every issuer and vendor that appears in the dataset,
  plus the whole pools (an ERP knows more parties than one month of POs uses).
- items: each issuer's code for every catalogue item (its item master).
- processed_po_numbers: a few hundred historical PO numbers per frequent issuer, plus the
  dataset POs marked expected_duplicate. Historical serials never overlap dataset serials
  (issuers.serial_range), so only the deliberate duplicates can match.
"""

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from datagen.generate import GeneratedPO
from datagen.india import CatalogueItem
from datagen.issuers import Counterparty, Issuer, serial_range

HISTORY_FROM = date(2023, 4, 1)  # historical POs: the two financial years before the dataset
HISTORY_TO = date(2025, 3, 31)
HISTORY_MIN, HISTORY_MAX = 200, 400  # historical PO numbers per frequent issuer


@dataclass(frozen=True)
class Masters:
    """The three master tables, as JSON-ready data."""

    parties: dict[str, dict[str, Any]]  # GSTIN -> {name, gstin, state_code, role, party_id}
    items: list[dict[str, Any]]  # one record per (issuer, item code)
    processed_po_numbers: dict[str, dict[str, Any]]  # issuer GSTIN -> {issuer_id, name, numbers}


def build_masters(
    pos: Sequence[GeneratedPO],
    issuers: Iterable[Issuer],
    counterparties: Iterable[Counterparty],
    catalogue: Sequence[CatalogueItem],
) -> Masters:
    """Build masters covering the given pools plus every party, item and issuer in `pos`."""
    all_issuers = _unique_issuers([*issuers, *(g.issuer for g in pos)])
    parties: dict[str, dict[str, Any]] = {}
    for issuer in all_issuers:
        for unit in issuer.units:
            _add_party(parties, unit.gstin, issuer.company.name, "buyer", issuer.id)
    for vendor in counterparties:
        _add_party(parties, vendor.unit.gstin, vendor.company.name, "vendor", vendor.id)
    for g in pos:  # covers anything the pools missed (e.g. a vendor from outside the pool)
        po = g.po
        _add_party(parties, po.vendor_gstin, po.vendor_name, "vendor", "")
        for gstin in (po.buyer_gstin, po.bill_to_gstin, po.ship_to_gstin):
            if gstin is not None:
                _add_party(parties, gstin, po.buyer_name, "buyer", g.meta.issuer_id)

    items = [
        {
            "issuer_id": issuer.id,
            "issuer_gstin": issuer.main.gstin,
            "item_code": issuer.item_code(item),
            "description": item.description,
            "hsn_sac": item.hsn_sac,
            "uom": item.uom,
        }
        for issuer in all_issuers
        for item in catalogue
    ]

    processed = {
        issuer.main.gstin: {
            "issuer_id": issuer.id,
            "issuer_name": issuer.company.name,
            "po_numbers": historical_po_numbers(issuer),
        }
        for issuer in all_issuers
        if issuer.frequent
    }
    for g in pos:
        if g.meta.expected_duplicate:
            entry = processed.setdefault(
                g.po.buyer_gstin,
                {"issuer_id": g.meta.issuer_id, "issuer_name": g.po.buyer_name, "po_numbers": []},
            )
            entry["po_numbers"] = sorted({*entry["po_numbers"], g.po.po_number})
    return Masters(parties, items, processed)


def historical_po_numbers(issuer: Issuer) -> list[str]:
    """A few hundred past PO numbers of one issuer, in its own format (stable per issuer)."""
    rng = random.Random(f"history:{issuer.id}")
    low, high = serial_range(issuer.po_number_style, historical=True)
    count = rng.randint(HISTORY_MIN, HISTORY_MAX)
    span = (HISTORY_TO - HISTORY_FROM).days
    numbers = {
        issuer.po_number(HISTORY_FROM + timedelta(days=rng.randint(0, span)), serial)
        for serial in rng.sample(range(low, high + 1), count)
    }
    return sorted(numbers)


def write_masters(masters: Masters, out_dir: Path) -> list[Path]:
    """Write parties.json, items.json and processed_po_numbers.json; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "parties.json": sorted(masters.parties.values(), key=lambda p: p["gstin"]),
        "items.json": masters.items,
        "processed_po_numbers.json": dict(sorted(masters.processed_po_numbers.items())),
    }
    paths = []
    for name, data in files.items():
        path = out_dir / name
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _unique_issuers(issuers: Iterable[Issuer]) -> list[Issuer]:
    by_id: dict[str, Issuer] = {}
    for issuer in issuers:
        by_id.setdefault(issuer.id, issuer)
    return list(by_id.values())


def _add_party(
    parties: dict[str, dict[str, Any]], gstin: str, name: str, role: str, party_id: str
) -> None:
    parties.setdefault(
        gstin,
        {"gstin": gstin, "name": name, "state_code": gstin[:2], "role": role, "party_id": party_id},
    )
