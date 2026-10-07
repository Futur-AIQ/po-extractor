"""Tests for baseline B0 (Step 2.2). LiteLLM is mocked with httpx.MockTransport: no network."""

import base64
import csv
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest
from PIL import Image

from baseline.pipeline import (
    BaselineSettings,
    build_request,
    extract_baseline,
    parse_po,
    render_pages,
    token_usage,
)
from baseline.run import run_baseline, select_docs
from datagen.build import MANIFEST_FIELDS
from datagen.generate import generate_po
from eval.dataset import load_dataset
from eval.run_format import RunConfig, RunWriter, read_output, read_raw, read_records
from eval.score import score_run

PO = generate_po(3).po
PO_JSON = PO.model_dump_json()


def _cfg(**overrides: Any) -> BaselineSettings:
    base = {"litellm_base_url": "http://litellm.test", "litellm_api_key": "sk-test",
            "model_alias": "po-baseline"}  # fmt: skip
    return BaselineSettings(**(base | overrides))


def _pdf(path: Path, pages: int) -> Path:
    with pymupdf.open() as doc:
        for number in range(1, pages + 1):
            doc.new_page(width=595, height=842).insert_text((72, 100), f"PURCHASE ORDER {number}")
        doc.save(path)
    return path


def _completion(content: str, finish: str = "stop", reasoning: str | None = None,
                usage: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": usage or {"prompt_tokens": 3000, "completion_tokens": 900},
    }


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _reply(payload: dict[str, Any], seen: list[dict[str, Any]] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append({"url": str(request.url), "auth": request.headers["authorization"],
                         "body": json.loads(request.content)})  # fmt: skip
        return httpx.Response(200, json=payload)

    return handler


# --- Rendering and the request --------------------------------------------------------------


@pytest.mark.parametrize("dpi", [72, 120])
def test_render_one_png_per_page_at_requested_dpi(tmp_path: Path, dpi: int) -> None:
    images = render_pages(_pdf(tmp_path / "po.pdf", 3), dpi)
    assert len(images) == 3
    for png in images:
        width, height = Image.open(io.BytesIO(png)).size
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert abs(width - 595 * dpi / 72) <= 1 and abs(height - 842 * dpi / 72) <= 1


def test_request_reproduces_todays_setup() -> None:
    body = build_request([b"png1", b"png2"], _cfg())
    assert body["model"] == "po-baseline" and body["max_tokens"] == 16000
    assert "temperature" not in body  # model default
    assert "chat_template_kwargs" not in body  # thinking at the model's default
    parts = body["messages"][1]["content"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert [base64.b64decode(p["image_url"]["url"].split(",")[1]) for p in images] == [
        b"png1",
        b"png2",
    ]
    assert not any("text layer" in str(p) for p in parts)
    schema = body["response_format"]["json_schema"]["schema"]
    assert {"total_tax", "po_number", "vendor_gstin"} <= set(schema["properties"])  # full names
    line_item = schema["$defs"]["LineItem"]
    assert line_item["type"] == "object" and "cgst_amount" in line_item["properties"]
    assert build_request([b"x"], _cfg(temperature=0.0))["temperature"] == 0.0


# --- Parsing and tokens ---------------------------------------------------------------------


def test_parse_po_accepts_fenced_json_and_rejects_truncated() -> None:
    assert parse_po(f"```json\n{PO_JSON}\n```", "stop") == (PO, None, None)
    po, data, error = parse_po(PO_JSON[:-40], "length")
    assert po is None and data is None
    assert "invalid JSON" in error and "finish_reason=length" in error
    assert parse_po("", "length") == (None, None, "empty content (finish_reason=length)")


def test_schema_invalid_json_keeps_its_values() -> None:
    # Seen on the local 9B: well-formed JSON, but "63 NOS" is not a decimal.
    bad = json.loads(PO_JSON)
    bad["line_items"][0]["quantity"] = "63 NOS"
    bad["subtotal"] = "56,49,775.24"
    po, data, error = parse_po(json.dumps(bad), "stop")
    assert po is None and error.startswith("schema validation: 2 errors")
    assert data["line_items"][0]["quantity"] == "63 NOS"


def test_reasoning_tokens_reported_or_estimated() -> None:
    reported = {"prompt_tokens": 10, "completion_tokens": 500,
                "completion_tokens_details": {"reasoning_tokens": 400}}  # fmt: skip
    usage = token_usage(reported, {"content": "{}", "reasoning_content": "x" * 99})
    assert (usage.reasoning, usage.reasoning_estimated) == (400, False)

    # llama.cpp: no count reported -> share of generated characters, flagged as an estimate
    plain = {"prompt_tokens": 10, "completion_tokens": 1000}
    usage = token_usage(plain, {"content": "y" * 250, "reasoning_content": "x" * 750})
    assert (usage.reasoning, usage.reasoning_estimated) == (750, True)
    # a reported 0 alongside reasoning text is not trusted
    zero = plain | {"completion_tokens_details": {"reasoning_tokens": 0}}
    assert token_usage(zero, {"content": "y", "reasoning_content": "x" * 3}).reasoning_estimated
    # reasoning left inside <think> in the content
    usage = token_usage(plain, {"content": "<think>" + "x" * 500 + "</think>" + "y" * 500})
    assert usage.reasoning == pytest.approx(500, abs=10) and usage.reasoning_estimated
    assert token_usage(plain, {"content": "{}"}).reasoning is None  # thinking off


# --- One document ---------------------------------------------------------------------------


async def test_extract_completed_with_reasoning_kept_in_raw(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []
    reply = _completion(PO_JSON, reasoning="Let me read page 1 ...")
    async with _client(_reply(reply, seen)) as client:
        result = await extract_baseline(_pdf(tmp_path / "po.pdf", 2), _cfg(), client)
    assert result.status == "completed" and result.po == PO and result.error is None
    assert set(result.timings_ms) == {"render", "llm", "parse", "total"}
    assert result.tokens.prompt == 3000 and result.tokens.completion == 900
    assert result.tokens.reasoning_estimated
    assert seen[0]["url"] == "http://litellm.test/v1/chat/completions"
    assert seen[0]["auth"] == "Bearer sk-test"
    response = result.raw["response"]["choices"][0]["message"]
    assert response["reasoning_content"] == "Let me read page 1 ..."
    assert "base64" not in json.dumps(result.raw)  # image data is not stored


async def test_truncated_json_is_a_parse_error_not_a_crash(tmp_path: Path) -> None:
    reply = _completion(PO_JSON[: len(PO_JSON) // 2], finish="length")
    async with _client(_reply(reply)) as client:
        result = await extract_baseline(_pdf(tmp_path / "po.pdf", 1), _cfg(), client)
    assert result.status == "parse_error" and result.po is None
    assert "finish_reason=length" in result.error
    assert result.raw["response"]["choices"][0]["finish_reason"] == "length"


async def test_http_error_and_timeout_are_failed_results(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "po.pdf", 1)
    async with _client(lambda request: httpx.Response(500, text="boom")) as client:
        result = await extract_baseline(pdf, _cfg(), client)
    assert result.status == "failed" and result.error.startswith("HTTP 500")

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with _client(timeout) as client:
        result = await extract_baseline(pdf, _cfg(), client)
    assert result.status == "failed" and "ReadTimeout" in result.error
    assert result.timings_ms["total"] >= result.timings_ms["llm"]


# --- A run on disk --------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Two POs with N and S variants: real (tiny) PDFs, truth = PO."""
    out = tmp_path / "synthetic"
    (out / "pdfs").mkdir(parents=True)
    (out / "truth").mkdir()
    rows = []
    for po_id in ("PO_0001", "PO_0002"):
        truth = {"po": PO.model_dump(mode="json"), "meta": {"id": po_id}}
        (out / "truth" / f"{po_id}.json").write_text(json.dumps(truth), encoding="utf-8")
        for variant, doc_id in (("N", po_id), ("S", f"{po_id}_S")):
            _pdf(out / "pdfs" / f"{doc_id}.pdf", 2)
            rows.append({"id": doc_id, "base_id": po_id, "variant": variant, "split": "dev",
                         "page_kinds": "native native", "layout": "L1_classic_erp",
                         "issuer_frequent": True, "pdf": f"pdfs/{doc_id}.pdf",
                         "truth": f"truth/{po_id}.json"})  # fmt: skip
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out


def test_limit_counts_pos_and_keeps_variants_paired(dataset: Path) -> None:
    docs = select_docs(dataset, "dev", ("N", "S"), limit=1)
    assert [d.doc_id for d in docs] == ["PO_0001", "PO_0001_S"]
    assert len(select_docs(dataset, "dev", ("S",), limit=None)) == 2


async def test_mocked_run_writes_standard_run_files(dataset: Path, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        parts = json.loads(request.content)["messages"][1]["content"]
        assert sum(p["type"] == "image_url" for p in parts) == 2  # both pages in one request
        # The second PO's scanned twin comes back truncated.
        calls.append(1)
        if len(calls) == 4:
            return httpx.Response(200, json=_completion(PO_JSON[:100], finish="length"))
        return httpx.Response(200, json=_completion(PO_JSON, reasoning="thinking..."))

    calls: list[int] = []
    docs = load_dataset(dataset, "dev", ("N", "S"))
    writer = RunWriter.create(tmp_path / "runs", "b0_test", RunConfig(pipeline="baseline"))
    lines: list[str] = []
    async with _client(handler) as client:
        records = await run_baseline(docs, _cfg(), writer, concurrency=1, client=client,
                                     progress=lines.append)  # fmt: skip

    run_dir = tmp_path / "runs" / "b0_test"
    on_disk = read_records(run_dir)
    assert on_disk == records and [r.doc_id for r in on_disk] == [d.doc_id for d in docs]
    assert [r.status for r in on_disk] == ["completed", "completed", "completed", "parse_error"]
    assert all(r.calls == 1 and r.tokens.completion == 900 for r in on_disk)
    assert read_output(run_dir, "PO_0001").po == PO.model_dump(mode="json")
    error = read_output(run_dir, "PO_0002_S")
    assert error.po is None and "invalid JSON" in error.error
    assert read_raw(run_dir, "PO_0002_S")["response"]["choices"][0]["finish_reason"] == "length"
    assert len(lines) == 4 and lines[0].startswith("[1/4] PO_0001") and "completion=900" in lines[0]
    assert lines[3].startswith("[4/4] PO_0002_S") and "invalid JSON" in lines[3]

    score = score_run(run_dir, dataset)  # the run format is what the scorer reads
    assert score.by_variant["N"].critical_doc_accuracy == 1.0
    assert score.by_variant["S"].critical_doc_accuracy == 0.5
    assert score.latency.parse_errors == 1 and score.latency.reasoning_estimated


async def test_unreadable_pdf_is_recorded_and_the_run_continues(
    dataset: Path, tmp_path: Path
) -> None:
    (dataset / "pdfs" / "PO_0001.pdf").write_bytes(b"not a pdf")
    docs = load_dataset(dataset, "dev", ("N",))
    writer = RunWriter.create(tmp_path / "runs", "r", RunConfig(pipeline="baseline"))
    async with _client(_reply(_completion(PO_JSON))) as client:
        records = await run_baseline(docs, _cfg(), writer, client=client, progress=lambda _: None)
    assert sorted(r.status for r in records) == ["completed", "failed"]


async def test_schema_invalid_output_is_a_parse_error_but_still_scored(
    dataset: Path, tmp_path: Path
) -> None:
    bad = json.loads(PO_JSON) | {"grand_total": "1,00,000.00 INR only"}
    docs = load_dataset(dataset, "dev", ("N",))[:1]
    writer = RunWriter.create(tmp_path / "runs", "r", RunConfig(pipeline="baseline"))
    async with _client(_reply(_completion(json.dumps(bad)))) as client:
        records = await run_baseline(docs, _cfg(), writer, client=client, progress=lambda _: None)
    assert records[0].status == "parse_error" and "grand_total" in records[0].error
    output = read_output(writer.run_dir, "PO_0001")
    assert output.status == "parse_error" and output.error == records[0].error
    assert output.po["grand_total"] == "1,00,000.00 INR only"

    score = score_run(writer.run_dir, dataset)
    errors = [(e.field, e.outcome.value) for e in score.errors]
    assert errors == [("grand_total", "wrong")]  # every other value read correctly still counts
    assert score.latency.parse_errors == 1
