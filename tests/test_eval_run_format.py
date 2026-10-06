"""Tests for the shared run format (Step 2.1)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from datagen.generate import generate_po
from eval.run_format import (
    DocRecord,
    RunConfig,
    RunWriter,
    TokenUsage,
    doc_id_for,
    read_config,
    read_output,
    read_raw,
    read_records,
    split_doc_id,
)


def test_doc_id_carries_the_variant() -> None:
    assert doc_id_for("PO_0001", "N") == "PO_0001"
    assert doc_id_for("PO_0001", "S") == "PO_0001_S"
    for doc_id, expected in [
        ("PO_0001", ("PO_0001", "N")),
        ("PO_0001_S", ("PO_0001", "S")),
        ("PO_0012_M", ("PO_0012", "M")),
    ]:
        assert split_doc_id(doc_id) == expected


def test_write_and_read_a_run(tmp_path: Path) -> None:
    config = RunConfig(pipeline="baseline", model_alias="po-baseline", dpi=120, thinking="default")
    writer = RunWriter.create(tmp_path, "r1", config)
    po = generate_po(1).po
    writer.write_output("PO_0001", po)
    writer.write_raw("PO_0001", [{"content": '{"po_no": "X"}', "usage": {"prompt_tokens": 9}}])
    writer.write_error("PO_0001_S", "parse_error", "Unterminated string at char 812")
    records = [
        DocRecord(doc_id="PO_0001", variant="N", status="completed", calls=1,
                  timings_ms={"total": 1500.0, "llm": {"H": 900.0}},
                  tokens=TokenUsage(prompt=4000, completion=800, reasoning=500)),
        DocRecord(doc_id="PO_0001_S", variant="S", status="parse_error", calls=1,
                  timings_ms={"total": 9000.0}, error="Unterminated string"),
    ]  # fmt: skip
    for record in records:
        writer.append_record(record)

    run_dir = tmp_path / "r1"
    assert read_config(run_dir) == config
    assert read_records(run_dir) == records
    output = read_output(run_dir, "PO_0001")
    assert output is not None and output.status == "completed"
    assert type(po).model_validate(output.po) == po  # full field names, Decimals exact
    error = read_output(run_dir, "PO_0001_S")
    assert error is not None and error.po is None and error.status == "parse_error"
    assert read_output(run_dir, "PO_0002") is None
    assert read_raw(run_dir, "PO_0001")[0]["usage"]["prompt_tokens"] == 9


def test_existing_run_is_kept_unless_overwrite(tmp_path: Path) -> None:
    RunWriter.create(tmp_path, "r1", RunConfig(pipeline="oracle")).write_raw("PO_0001", {})
    with pytest.raises(FileExistsError):
        RunWriter.create(tmp_path, "r1", RunConfig(pipeline="oracle"))
    RunWriter.create(tmp_path, "r1", RunConfig(pipeline="oracle"), overwrite=True)
    assert not (tmp_path / "r1" / "raw" / "PO_0001.json").exists()

    (tmp_path / "not_a_run").mkdir()
    with pytest.raises(FileExistsError, match="not a run directory"):
        RunWriter.create(tmp_path, "not_a_run", RunConfig(pipeline="oracle"), overwrite=True)


def test_record_needs_total_timing_and_known_status() -> None:
    with pytest.raises(ValidationError, match="total"):
        DocRecord(doc_id="PO_0001", variant="N", status="completed", timings_ms={"llm": 1.0})
    with pytest.raises(ValidationError):
        DocRecord(doc_id="PO_0001", variant="N", status="ok", timings_ms={"total": 1.0})
