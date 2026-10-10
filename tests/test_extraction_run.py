"""Smoke test for the optimised pipeline CLI (Step 3.8): one document, a fake LLM client.

Builds a one-PO dataset in a temp folder (generated PO, two-page text PDF, manifest, masters),
runs `app.extraction.run.main`, and checks the run folder with the eval readers and scorer.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from app.extraction.llm_client import CallResult, Timings, Tokens
from app.extraction.run import main
from datagen.build import MANIFEST_FIELDS
from datagen.generate import Knobs, generate_po
from eval.run_format import read_config, read_output, read_raw, read_records
from eval.score import score_run
from schema.fields import COMPUTED_FIELDS, LLM_LINE_ITEM_COLUMNS

PO = generate_po(23, Knobs(min_lines=6, max_lines=6, typical_min_lines=6, typical_max_lines=6)).po
PAGE_ROWS = {1: [1, 2, 3], 2: [4, 5, 6]}


def write_pdf(path: Path) -> None:
    """Two native pages with the identifiers the grounding rule looks for."""
    gstins = (PO.buyer_gstin, PO.vendor_gstin, PO.bill_to_gstin, PO.ship_to_gstin)
    head = [f"PURCHASE ORDER {PO.po_number}", *(g for g in gstins if g)]
    head += [p for p in (PO.buyer_pan, PO.vendor_pan) if p]
    with pymupdf.open() as doc:
        for page_no, numbers in PAGE_ROWS.items():
            lines = head if page_no == 1 else [f"Page {page_no}"]
            for item in PO.line_items:
                if item.line_no in numbers:
                    lines += [str(item.line_no), item.item_code or "", "Item", item.hsn_sac]
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), "\n".join(lines), fontsize=8)
        doc.save(path)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    out = tmp_path / "synthetic"
    for folder in ("pdfs", "truth", "masters"):
        (out / folder).mkdir(parents=True)
    write_pdf(out / "pdfs" / "PO_0001.pdf")
    truth = {"po": PO.model_dump(mode="json"), "meta": {"id": "PO_0001"}}
    (out / "truth" / "PO_0001.json").write_text(json.dumps(truth), encoding="utf-8")
    row = {
        "id": "PO_0001",
        "base_id": "PO_0001",
        "variant": "N",
        "split": "dev",
        "page_kinds": "native native",
        "layout": "L1_classic_erp",
        "issuer_frequent": True,
        "pdf": "pdfs/PO_0001.pdf",
        "truth": "truth/PO_0001.json",
    }
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, restval="")
        writer.writeheader()
        writer.writerow(row)
    gstins = [g for g in (PO.buyer_gstin, PO.vendor_gstin, PO.bill_to_gstin, PO.ship_to_gstin) if g]
    masters = {
        "parties.json": [{"gstin": g} for g in gstins],
        "items.json": [
            {"issuer_gstin": PO.buyer_gstin, "item_code": i.item_code}
            for i in PO.line_items
            if i.item_code
        ],
        "processed_po_numbers.json": {},
    }
    for name, data in masters.items():
        (out / "masters" / name).write_text(json.dumps(data), encoding="utf-8")
    return out


class FakeClient:
    """Answers every call with the PO's own values, as a perfect model would."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None]] = []

    async def call(self, alias, messages, schema, max_tokens, priority=None, call_id=None):
        self.calls.append((alias, call_id, priority))
        if call_id == "H":
            parsed: dict[str, Any] = PO.model_dump(
                mode="json", exclude={"line_items", *COMPUTED_FIELDS}, exclude_none=True
            )
        else:
            numbers = None if call_id == "LI" else PAGE_ROWS[int(call_id.removeprefix("LI-p"))]
            lines = PO.model_dump(mode="json")["line_items"]
            parsed = {
                "rows": [
                    [line[c] for c in LLM_LINE_ITEM_COLUMNS]
                    for line in lines
                    if numbers is None or line["line_no"] in numbers
                ]
            }
        now = time.time()
        return CallResult(
            call_id,
            alias,
            parsed,
            json.dumps(parsed),
            None,
            Tokens(prompt=1200, completion=80),
            Timings(now, now, 3.0),
            1,
        )


def test_cli_writes_a_valid_run_folder(dataset: Path, tmp_path: Path) -> None:
    client = FakeClient()
    runs = tmp_path / "runs"
    argv = [
        "--split",
        "dev",
        "--variants",
        "N",
        "--run-id",
        "smoke",
        "--concurrency",
        "2",
        "--dataset",
        str(dataset),
        "--runs-dir",
        str(runs),
    ]
    records = main(argv, client=client)
    run_dir = runs / "smoke"

    config = read_config(run_dir)
    assert config.pipeline == "optimised" and config.model_alias == "po-fast"
    assert config.strategy == "adaptive" and config.extra["variants"] == ["N"]
    assert config.extra["dataset"] == str(dataset)

    [record] = read_records(run_dir)
    assert record == records[0]
    assert record.doc_id == "PO_0001" and record.status == "completed"
    assert record.calls == 2 and record.tokens.prompt == 2400 and record.tokens.completion == 160
    assert {"preprocess", "llm", "validate", "total"} <= set(record.timings_ms)
    assert record.routing["strategy"] == "two_call" and record.routing["retried_calls"] == []

    output = read_output(run_dir, "PO_0001")
    assert output.status == "completed" and output.po["po_number"] == PO.po_number
    assert output.po["vendor_gstin"] == PO.vendor_gstin  # full field names, not short keys
    assert len(output.po["line_items"]) == 6

    raw = read_raw(run_dir, "PO_0001")
    assert [c["call_id"] for c in raw["calls"]] == ["H", "LI"]
    assert raw["envelope"]["validation"]["passed"] is True
    assert {priority for _, _, priority in client.calls} == {1}  # first (and only) PO

    # The eval scorer reads the folder like any other run.
    score = score_run(run_dir, dataset)
    assert score.overall.docs == 1 and score.overall.critical_doc_accuracy == 1.0


def test_ablation_flags_reach_the_pipeline(dataset: Path, tmp_path: Path) -> None:
    client = FakeClient()
    argv = [
        "--variants",
        "N",
        "--run-id",
        "ablation",
        "--dataset",
        str(dataset),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--strategy",
        "per_page",
        "--dpi",
        "72",
        "--native-mode",
        "text",
        "--no-skip-pages",
        "--no-retry",
    ]
    [record] = main(argv, client=client)
    config = read_config(tmp_path / "runs" / "ablation")
    assert (config.strategy, config.dpi) == ("per_page", 72)
    assert config.extra["native_mode"] == "text" and config.extra["max_reasks"] == 0
    assert config.extra["skip_fieldless_pages"] is False
    assert record.routing["strategy"] == "per_page"
    assert [call_id for _, call_id, _ in client.calls] == ["H", "LI-p1", "LI-p2"]
