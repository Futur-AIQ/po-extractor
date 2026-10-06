"""Tests for the scoring engine (Step 2.1). No LLM: predictions are made by hand."""

import copy
import csv
import json
import random
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from datagen.build import MANIFEST_FIELDS
from datagen.generate import generate_many, generate_po
from eval.dataset import DatasetDoc
from eval.run_format import DocRecord, RunConfig, RunWriter, TokenUsage
from eval.score import (
    GroupSummary,
    Outcome,
    aggregate_run,
    match_rows,
    percentile,
    score_document,
    score_run,
    summarize_latency,
)
from schema.fields import CRITICAL_HEADER_FIELDS, HEADER_FIELDS
from scripts.make_oracle_run import make_oracle_run


def _doc(doc_id: str = "PO_0001", variant: str = "N", layout: str = "L1_classic_erp") -> DatasetDoc:
    base = doc_id.removesuffix(f"_{variant}")
    return DatasetDoc(doc_id, base, variant, "test", layout, True, ("native",), Path(), Path())


def _truth(seed: int = 5) -> dict[str, Any]:
    return generate_po(seed).po.model_dump(mode="json")


def _other_char(c: str) -> str:
    return "A" if c != "A" else "B"


def _assert_perfect(summary: GroupSummary) -> None:
    assert summary.critical_doc_accuracy == 1.0
    assert summary.field_accuracy == 1.0
    assert summary.header.accuracy == 1.0 and summary.cells.accuracy == 1.0
    assert summary.row_recall == summary.row_precision == summary.row_f1 == 1.0
    for tally in (summary.header, summary.cells):
        assert tally.outcomes[Outcome.WRONG] == tally.outcomes[Outcome.MISSED] == 0
        assert tally.outcomes[Outcome.HALLUCINATED] == 0
        assert tally.text_exact == tally.text_fuzzy == tally.text_total


# --- One document -----------------------------------------------------------------------------


def test_truth_against_itself_is_perfect() -> None:
    truth = _truth()
    score = score_document(_doc(), truth, copy.deepcopy(truth))
    assert score.critical_correct and score.errors() == []
    assert all(m.by_line_no for m in score.matches)
    summary = GroupSummary()
    summary.add(score)
    _assert_perfect(summary)
    assert summary.header.total == len(HEADER_FIELDS)
    assert summary.cells.total == len(truth["line_items"]) * 15


def test_one_wrong_one_missed_one_hallucinated_one_dropped_row() -> None:
    truth = _truth()
    pred = copy.deepcopy(truth)
    gstin = truth["vendor_gstin"]
    pred["vendor_gstin"] = gstin[:-1] + _other_char(gstin[-1])  # wrong
    missed = next(f for f in HEADER_FIELDS if f not in CRITICAL_HEADER_FIELDS and truth[f])
    del pred[missed]  # missed
    hallucinated = next(f for f in HEADER_FIELDS if truth[f] is None)
    pred[hallucinated] = "X-123"  # hallucinated
    dropped = pred["line_items"].pop(3)  # dropped row (line_no 4)

    score = score_document(_doc(), truth, pred)
    summary = GroupSummary()
    summary.add(score)
    header = summary.header.outcomes
    assert (header[Outcome.WRONG], header[Outcome.MISSED], header[Outcome.HALLUCINATED]) == (
        1,
        1,
        1,
    )
    assert summary.cells.accuracy == 1.0  # every remaining row matched by line_no, all correct
    n = len(truth["line_items"])
    assert (summary.truth_rows, summary.pred_rows, summary.matched_rows) == (n, n - 1, n - 1)
    assert summary.row_recall == (n - 1) / n and summary.row_precision == 1.0
    assert not score.critical_correct

    errors = {(e.field, e.outcome, e.row) for e in score.errors()}
    assert errors == {
        ("vendor_gstin", Outcome.WRONG, None),
        (missed, Outcome.MISSED, None),
        (hallucinated, Outcome.HALLUCINATED, None),
        ("line_items", Outcome.MISSED, dropped["line_no"]),
    }


def test_formatting_differences_are_not_errors() -> None:
    truth = _truth()
    pred = copy.deepcopy(truth)
    pred["grand_total"] = f"{float(truth['grand_total']):,}"  # 6231123.0 -> '6,231,123.0'
    pred["subtotal"] = float(truth["subtotal"])
    pred["vendor_gstin"] = truth["vendor_gstin"].lower()
    pred["buyer_name"] = truth["buyer_name"].upper() + "."
    year, month, day = truth["po_date"].split("-")
    pred["po_date"] = f"{day}/{month}/{year}"
    pred["line_items"][0]["quantity"] = f"{Decimal(truth['line_items'][0]['quantity']):.3f}"
    assert score_document(_doc(), truth, pred).errors() == []


def test_indian_grouping_matches_plain_decimal() -> None:
    truth = _truth() | {"grand_total": "123456.50"}
    pred = truth | {"grand_total": "1,23,456.50"}
    assert score_document(_doc(), truth, pred).critical_correct
    pred = truth | {"grand_total": 123456.5}
    assert score_document(_doc(), truth, pred).critical_correct


def test_fuzzy_text_scored_separately_from_exact() -> None:
    truth = _truth()
    pred = copy.deepcopy(truth)
    pred["payment_terms"] = truth["payment_terms"] + " (net)"  # near miss
    score = score_document(_doc(), truth, pred)
    result = next(r for r in score.header if r.field == "payment_terms")
    assert result.outcome is Outcome.WRONG and result.fuzzy is not None and result.fuzzy >= 90
    summary = GroupSummary()
    summary.add(score)
    assert summary.header.text_exact == summary.header.text_total - 1
    assert summary.header.text_fuzzy == summary.header.text_total


def test_rows_without_line_no_fall_back_to_content_matching() -> None:
    truth = _truth()
    rows = copy.deepcopy(truth["line_items"])
    for row in rows:
        row["line_no"] = None
    random.Random(1).shuffle(rows)
    matches = match_rows(truth["line_items"], rows)
    assert len(matches) == len(rows) and not any(m.by_line_no for m in matches)
    for m in matches:  # each truth row found its own (shuffled) prediction
        assert (
            rows[m.pred_index]["description"] == truth["line_items"][m.truth_index]["description"]
        )

    # Duplicate line numbers are not usable either; content decides.
    dup = copy.deepcopy(truth["line_items"][:3])
    dup[1]["line_no"] = dup[0]["line_no"]
    matches = match_rows(truth["line_items"][:3], dup)
    assert [(m.truth_index, m.pred_index) for m in matches] == [(0, 0), (1, 1), (2, 2)]


def test_unrelated_rows_are_not_matched() -> None:
    truth = _truth()["line_items"][:2]
    other = [
        {"line_no": None, "item_code": "ZZ-1", "description": "Unrelated widget", "quantity": "999"}
    ]
    assert match_rows(truth, other) == []


def test_computed_fields_can_be_excluded() -> None:
    truth = _truth()
    pred = copy.deepcopy(truth)
    pred["total_tax"] = "1.00"
    for row in pred["line_items"]:
        row["cgst_amount"] = row["sgst_amount"] = row["igst_amount"] = None
    included = score_document(_doc(), truth, pred)
    assert any(e.field == "total_tax" for e in included.errors())
    excluded = score_document(_doc(), truth, pred, include_computed=False)
    assert excluded.errors() == []
    assert {r.field for r in excluded.cells}.isdisjoint({"cgst_amount", "igst_amount"})


def test_no_output_scores_as_all_missed() -> None:
    truth = _truth()
    score = score_document(_doc(), truth, None, status="parse_error")
    assert not score.critical_correct and score.matches == []
    assert all(r.outcome in (Outcome.MISSED, Outcome.CORRECT_ABSENT) for r in score.header)


# --- Aggregation ------------------------------------------------------------------------------


def test_aggregation_by_variant_layout_and_issuer() -> None:
    truth = _truth()
    wrong = copy.deepcopy(truth) | {"grand_total": "1.00"}
    scores = [
        score_document(_doc("PO_0001", "N", "L1_classic_erp"), truth, copy.deepcopy(truth)),
        score_document(_doc("PO_0001_S", "S", "L1_classic_erp"), truth, wrong),
        score_document(_doc("PO_0002", "N", "L2_tally_style"), truth, copy.deepcopy(truth)),
        score_document(_doc("PO_0002_S", "S", "L2_tally_style"), truth, copy.deepcopy(truth)),
    ]
    records = [
        DocRecord(doc_id=s.doc.doc_id, variant=s.doc.variant, status="completed",
                  timings_ms={"total": 1000.0 * (i + 1)})
        for i, s in enumerate(scores)
    ]  # fmt: skip
    run = aggregate_run(scores, records)
    assert run.overall.docs == 4 and run.overall.critical_doc_accuracy == 0.75
    assert run.by_variant["N"].critical_doc_accuracy == 1.0
    assert run.by_variant["S"].critical_doc_accuracy == 0.5
    assert run.by_variant["S"].header.outcomes[Outcome.WRONG] == 1
    assert run.by_layout["L1_classic_erp"].critical_doc_accuracy == 0.5
    assert run.by_layout["L2_tally_style"].critical_doc_accuracy == 1.0
    assert list(run.by_issuer) == ["frequent"]
    assert [(e.doc_id, e.variant, e.field) for e in run.errors] == [
        ("PO_0001_S", "S", "grand_total")
    ]
    assert run.latency_by_variant["S"].max_ms == 4000.0
    json.dumps(run.as_dict())  # JSON-able


def test_latency_and_token_summary() -> None:
    records = [
        DocRecord(doc_id=f"PO_{i:04d}", variant="N", status="completed", calls=2,
                  timings_ms={"total": float(ms)},
                  tokens=TokenUsage(prompt=1000, completion=100 * i,
                                    reasoning=50 if i % 2 else None))
        for i, ms in enumerate([1000, 2000, 3000, 4000, 10000], start=1)
    ]  # fmt: skip
    records.append(
        DocRecord(doc_id="PO_0006", variant="S", status="parse_error", timings_ms={"total": 5000})
    )
    summary = summarize_latency(records)
    assert summary.docs == 6 and summary.parse_errors == 1
    assert summary.p50_ms == 3500.0 and summary.max_ms == 10000.0
    assert summary.p95_ms == pytest.approx(8750.0)
    assert summary.mean_completion_tokens == pytest.approx(1500 / 6)
    assert summary.mean_reasoning_tokens == 50  # only docs that reported it
    assert percentile([], 95) is None and percentile([7.0], 95) == 7.0


# --- Whole runs on disk -----------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A tiny dataset on disk: manifest + truth for 3 POs with N and S variants (no PDFs)."""
    out = tmp_path / "synthetic"
    (out / "truth").mkdir(parents=True)
    rows = []
    for i, generated in enumerate(generate_many(3, seed=11), start=1):
        po_id = f"PO_{i:04d}"
        truth = {"po": generated.po.model_dump(mode="json"), "meta": {"id": po_id}}
        (out / "truth" / f"{po_id}.json").write_text(json.dumps(truth), encoding="utf-8")
        for variant, doc_id in (("N", po_id), ("S", f"{po_id}_S")):
            rows.append({
                "id": doc_id, "base_id": po_id, "variant": variant, "page_kinds": "native",
                "layout": generated.meta.layout, "issuer_frequent": generated.meta.issuer_frequent,
                "split": "dev" if i == 1 else "test", "pdf": f"pdfs/{doc_id}.pdf",
                "truth": f"truth/{po_id}.json",
            })  # fmt: skip
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out


def test_oracle_run_scores_100_percent_everywhere(dataset: Path, tmp_path: Path) -> None:
    run_dir = make_oracle_run(dataset, tmp_path / "runs", split=None)
    assert run_dir.name == "oracle_all"
    run = score_run(run_dir, dataset)
    assert run.overall.docs == 6 and run.errors == []
    for summary in [run.overall, *run.by_variant.values(), *run.by_layout.values(),
                    *run.by_issuer.values()]:  # fmt: skip
        _assert_perfect(summary)
    assert set(run.by_variant) == {"N", "S"}

    dev = make_oracle_run(dataset, tmp_path / "runs", split="dev", variants=("S",))
    assert [s.doc.doc_id for s in score_run(dev, dataset).docs] == ["PO_0001_S"]


def test_score_run_counts_parse_errors_and_rejects_unknown_docs(
    dataset: Path, tmp_path: Path
) -> None:
    writer = RunWriter.create(tmp_path / "runs", "r", RunConfig(pipeline="test"))
    writer.write_error("PO_0001_S", "parse_error", "truncated JSON")
    writer.append_record(DocRecord(doc_id="PO_0001_S", variant="S", status="parse_error",
                                   timings_ms={"total": 30000.0}))  # fmt: skip
    run = score_run(writer.run_dir, dataset)
    assert run.overall.critical_doc_accuracy == 0.0 and run.overall.row_recall == 0.0
    assert run.overall.row_precision is None  # no predicted rows: n/a
    assert run.latency.parse_errors == 1 and run.overall.statuses["parse_error"] == 1

    writer.append_record(DocRecord(doc_id="PO_9999", variant="N", status="completed",
                                   timings_ms={"total": 1.0}))  # fmt: skip
    with pytest.raises(ValueError, match="PO_9999"):
        score_run(writer.run_dir, dataset)
