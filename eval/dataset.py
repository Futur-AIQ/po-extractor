"""Read the synthetic dataset built in Phase 1 (data/synthetic) for runs and scoring.

One manifest row per document variant: PO_0001 (native), PO_0001_S (scanned twin),
PO_0001_M (mixed). All variants of a PO share one truth file.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VARIANTS = ("N", "S", "M")


@dataclass(frozen=True)
class DatasetDoc:
    """One document variant of the dataset, as listed in manifest.csv."""

    doc_id: str  # PO_0001, PO_0001_S, PO_0001_M
    base_id: str  # PO_0001
    variant: str  # N, S or M
    split: str  # dev or test
    layout: str
    issuer_frequent: bool
    page_kinds: tuple[str, ...]  # per page: native / scanned
    pdf: Path
    truth: Path


def load_dataset(
    dataset_dir: Path, split: str | None = None, variants: tuple[str, ...] = VARIANTS
) -> list[DatasetDoc]:
    """Documents of the dataset in manifest order, optionally limited to a split and variants."""
    with (dataset_dir / "manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    docs = [
        DatasetDoc(
            doc_id=row["id"],
            base_id=row["base_id"],
            variant=row["variant"],
            split=row["split"],
            layout=row["layout"],
            issuer_frequent=row["issuer_frequent"] == "True",
            page_kinds=tuple(row["page_kinds"].split()),
            pdf=dataset_dir / row["pdf"],
            truth=dataset_dir / row["truth"],
        )
        for row in rows
    ]
    return [
        doc for doc in docs if (split is None or doc.split == split) and doc.variant in variants
    ]


def select_docs(
    dataset_dir: Path, split: str | None, variants: tuple[str, ...], limit: int | None
) -> list[DatasetDoc]:
    """Documents of the split in manifest order, limited to the first `limit` POs."""
    docs = load_dataset(dataset_dir, split, variants)
    if limit is None:
        return docs
    base_ids = list(dict.fromkeys(doc.base_id for doc in docs))[:limit]
    return [doc for doc in docs if doc.base_id in base_ids]


def read_truth_po(doc: DatasetDoc) -> dict[str, Any]:
    """The ground-truth PO of a document, as JSON (full field names, Decimals as strings)."""
    return json.loads(doc.truth.read_text(encoding="utf-8"))["po"]
