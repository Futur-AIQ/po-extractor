"""Tests for the evaluation report and run comparison (Step 2.3). No LLM."""

import csv
import json
import shutil
from pathlib import Path

import pytest

from datagen.build import MANIFEST_FIELDS
from datagen.generate import generate_many
from eval.compare import compare_runs, render_markdown
from eval.report import top_problem_fields, write_report
from eval.run_format import read_output
from eval.score import ErrorRecord, Outcome
from scripts.make_oracle_run import make_oracle_run


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Manifest + truth for 3 POs with N and S variants (no PDFs needed for scoring)."""
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


@pytest.fixture
def oracle(dataset: Path, tmp_path: Path) -> Path:
    return make_oracle_run(dataset, tmp_path / "runs", split=None)


def _degrade(run_dir: Path, target: Path, doc_ids: list[str], change) -> Path:
    """Copy of a run where `change(po)` is applied to the outputs of `doc_ids`."""
    shutil.copytree(run_dir, target)
    for doc_id in doc_ids:
        output = read_output(target, doc_id)
        change(output.po)
        (target / "outputs" / f"{doc_id}.json").write_text(output.model_dump_json(), "utf-8")
    return target


def _wrong_grand_total(po: dict) -> None:
    po["grand_total"] = "1.00"


def _drop_codes_and_units(po: dict) -> None:  # non-critical columns only
    for row in po["line_items"]:
        row["item_code"] = row["uom"] = None


# --- Report -----------------------------------------------------------------------------------


def test_oracle_report_is_100_percent(oracle: Path) -> None:
    score = write_report(oracle)  # dataset taken from config.json
    report = json.loads((oracle / "report.json").read_text())
    groups = [report["overall"], *report["by_variant"].values(), *report["by_layout"].values(),
              *report["by_issuer"].values()]  # fmt: skip
    for group in groups:
        assert group["critical_doc_accuracy"] == group["field_accuracy"] == 1.0
        assert group["line_items"]["recall"] == group["line_items"]["precision"] == 1.0
    assert set(report["by_variant"]) == {"N", "S"} and report["errors"] == 0
    assert report["latency"]["parse_errors"] == 0 and report["top_problem_fields"] == []
    assert report["config"]["pipeline"] == "oracle"

    markdown = (oracle / "report.md").read_text()
    assert "| overall | 6 | 100.0% | 100.0%" in markdown
    assert "**no**" not in markdown and "No errors." in markdown
    assert (oracle / "errors.csv").read_text().splitlines() == [
        "doc_id,variant,field,outcome,row,truth,pred,fuzzy"
    ]
    assert score.overall.docs == 6


def test_report_lists_errors_by_field_frequency(oracle: Path, tmp_path: Path) -> None:
    def change(po: dict) -> None:
        _wrong_grand_total(po)
        po["vendor_gstin"] = "29AAAAA0000A1Z5"

    run = _degrade(oracle, tmp_path / "bad", ["PO_0001", "PO_0002_S"], change)
    run_score = write_report(run)
    rows = list(csv.DictReader((run / "errors.csv").open()))
    assert len(rows) == len(run_score.errors) == 4  # 2 fields x 2 documents
    counts = [sum(r["field"] == row["field"] for r in rows) for row in rows]
    assert counts == sorted(counts, reverse=True)  # most frequent field first
    report = json.loads((run / "report.json").read_text())
    problems = {p["field"]: p for p in report["top_problem_fields"]}
    assert problems["grand_total"]["errors"] == 2
    examples = problems["grand_total"]["examples"]
    assert {e["doc_id"] for e in examples} == {"PO_0001", "PO_0002_S"}  # from different docs
    assert all(e["pred"] == "1.00" for e in examples)
    assert report["by_variant"]["N"]["critical_doc_accuracy"] == pytest.approx(2 / 3)
    assert "**no**" in (run / "report.md").read_text()


def test_top_problem_fields_limits_and_examples() -> None:
    errors = [
        ErrorRecord(f"PO_{d:04d}", "N", field, Outcome.WRONG, "a", "b")
        for field, docs in [("x", 3), ("y", 5), ("z", 1)]
        for d in range(docs)
    ]
    problems = top_problem_fields(errors, n=2, examples=2)
    assert [(p.field, p.errors) for p in problems] == [("y", 5), ("x", 3)]
    assert all(len(p.examples) == 2 for p in problems)


# --- Compare ----------------------------------------------------------------------------------


def test_compare_with_itself_accepts_with_zero_deltas(oracle: Path) -> None:
    comparison = compare_runs(oracle, oracle)
    assert comparison.decision.accept and comparison.decision.verdict == "ACCEPT"
    assert comparison.common_docs == 6
    for row in comparison.overall_rows() + comparison.variant_rows():
        assert row.delta == 0, row.name
    assert "critical-doc accuracy unchanged" in comparison.decision.reasons[0]
    assert "**ACCEPT**" in render_markdown(comparison)


def test_wrong_critical_value_rejects_for_critical_doc_accuracy(
    oracle: Path, tmp_path: Path
) -> None:
    degraded = _degrade(oracle, tmp_path / "bad", ["PO_0001_S"], _wrong_grand_total)
    decision = compare_runs(oracle, degraded).decision
    assert not decision.accept
    assert decision.critical_doc_drop_pp == pytest.approx(100 / 6)
    assert decision.field_drop_pp < 1  # one value out of thousands
    assert decision.reasons == ["critical-doc accuracy drops 16.67 pp (exceeds the 0.5 pp limit)"]


def test_many_non_critical_errors_reject_for_field_accuracy(oracle: Path, tmp_path: Path) -> None:
    ids = ["PO_0001", "PO_0001_S", "PO_0002", "PO_0002_S", "PO_0003", "PO_0003_S"]
    degraded = _degrade(oracle, tmp_path / "bad", ids, _drop_codes_and_units)
    comparison = compare_runs(oracle, degraded)
    decision = comparison.decision
    assert not decision.accept and decision.critical_doc_drop_pp == 0
    assert len(decision.reasons) == 1 and decision.reasons[0].startswith("field accuracy drops")
    assert "exceeds the 1 pp limit" in decision.reasons[0]
    markdown = render_markdown(comparison)
    assert "**REJECT**: field accuracy drops" in markdown


def test_compare_uses_only_common_documents(dataset: Path, tmp_path: Path, oracle: Path) -> None:
    dev = make_oracle_run(dataset, tmp_path / "runs", split="dev", variants=("N",))
    comparison = compare_runs(oracle, dev)
    assert (comparison.common_docs, comparison.ref_only, comparison.cand_only) == (1, 5, 0)
    assert comparison.ref.overall.docs == comparison.cand.overall.docs == 1


def test_small_drop_within_limits_is_accepted(oracle: Path, tmp_path: Path) -> None:
    def blank_terms(po: dict) -> None:  # one non-critical header field
        po["payment_terms"] = None if po["payment_terms"] else "Net 30"

    degraded = _degrade(oracle, tmp_path / "ok", ["PO_0001"], blank_terms)
    decision = compare_runs(oracle, degraded).decision
    assert decision.accept and 0 < decision.field_drop_pp < 1
    assert decision.reasons[1].startswith("field accuracy drops")
