"""Run baseline B0 over a dataset split and write the standard run format (eval/run_format.py).

    uv run python -m baseline.run --split dev --variants N,S --limit 3 --concurrency 1 \\
        --run-id b0_dev_local

Needs the local model and LiteLLM running (`PARALLEL=1 CTX_SIZE=32768 make model`,
`make litellm`). --limit counts POs: each selected PO runs in every requested variant, so
native and scanned results stay paired. One progress line is printed per document.
"""

import argparse
import asyncio
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from app.core.settings import get_settings
from baseline.pipeline import (
    DEFAULT_DPI,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_S,
    BaselineResult,
    BaselineSettings,
    extract_baseline,
)
from eval.dataset import VARIANTS, DatasetDoc, select_docs
from eval.run_format import (
    DocRecord,
    RunConfig,
    RunWriter,
    TokenUsage,
    current_git_commit,
    new_run_id,
)


def save_result(writer: RunWriter, doc: DatasetDoc, result: BaselineResult) -> DocRecord:
    """Write the output (or error record), raw response and record line of one document."""
    if result.po is not None:
        writer.write_output(doc.doc_id, result.po, result.status)
    elif result.invalid_data is not None:  # parse_error, but the values are scored
        writer.write_output(doc.doc_id, result.invalid_data, result.status, result.error)
    else:
        writer.write_error(doc.doc_id, result.status, result.error or "unknown error")
    writer.write_raw(doc.doc_id, result.raw)
    record = DocRecord(
        doc_id=doc.doc_id,
        variant=doc.variant,
        status=result.status,
        timings_ms=result.timings_ms,
        tokens=result.tokens,
        calls=result.calls,
        error=result.error,
    )
    writer.append_record(record)
    return record


def _crashed(error: Exception, seconds: float) -> BaselineResult:
    """A result for an unexpected exception (e.g. an unreadable PDF): the run goes on."""
    message = f"{type(error).__name__}: {error}"
    timings = {"total": round(seconds * 1000, 1)}
    return BaselineResult(None, "failed", message, {"error": message}, timings, TokenUsage(), 0)


def progress_line(index: int, total: int, record: DocRecord) -> str:
    seconds = record.total_ms / 1000
    reasoning, estimated = record.tokens.reasoning, record.tokens.reasoning_estimated
    thinking = "" if reasoning is None else f" reasoning{'~' if estimated else '='}{reasoning}"
    error = f"  {record.error[:120]}" if record.error else ""
    return (
        f"[{index}/{total}] {record.doc_id:<10} {record.status:<11} {seconds:7.1f}s "
        f"completion={record.tokens.completion}{thinking}{error}"
    )


async def run_baseline(
    docs: list[DatasetDoc],
    cfg: BaselineSettings,
    writer: RunWriter,
    concurrency: int = 1,
    client: httpx.AsyncClient | None = None,
    progress: Callable[[str], None] = print,
) -> list[DocRecord]:
    """Extract every document with at most `concurrency` in flight; returns the records."""
    semaphore = asyncio.Semaphore(concurrency)
    done: list[DocRecord] = []

    async def one(doc: DatasetDoc, http: httpx.AsyncClient) -> None:
        async with semaphore:
            start = time.perf_counter()
            try:
                result = await extract_baseline(doc.pdf, cfg, http)
            except Exception as exc:  # never let one document stop the run
                result = _crashed(exc, time.perf_counter() - start)
            record = await asyncio.to_thread(save_result, writer, doc, result)
            done.append(record)
            progress(progress_line(len(done), len(docs), record))

    if client is not None:
        await asyncio.gather(*(one(doc, client) for doc in docs))
    else:
        async with httpx.AsyncClient() as own_client:
            await asyncio.gather(*(one(doc, own_client) for doc in docs))
    return done


def main() -> None:
    """CLI entry point."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run baseline B0 (today's approach).")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--variants", default="N,S", help=f"subset of {','.join(VARIANTS)}")
    parser.add_argument("--limit", type=int, default=None, help="number of POs (default: all)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--run-id", default=None, help="default: baseline_<timestamp>")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    parser.add_argument("--temperature", type=float, default=None, help="default: not sent")
    parser.add_argument("--model", default=settings.po_baseline, help="LiteLLM alias")
    parser.add_argument("--dataset", type=Path, default=Path(settings.data_dir) / "synthetic")
    parser.add_argument("--runs-dir", type=Path, default=Path(settings.runs_dir))
    args = parser.parse_args()

    if not settings.litellm_api_key.get_secret_value():
        sys.exit("LITELLM_API_KEY is not set in .env (see .env.example)")
    split = None if args.split == "all" else args.split
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    docs = select_docs(args.dataset, split, variants, args.limit)
    if not docs:
        sys.exit(f"no documents for split={args.split} variants={args.variants}")

    cfg = BaselineSettings.from_settings(
        settings,
        model_alias=args.model,
        dpi=args.dpi,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        temperature=args.temperature,
    )
    config = RunConfig(
        pipeline="baseline",
        model_alias=cfg.model_alias,
        strategy="single_call",
        dpi=cfg.dpi,
        thinking="default",
        git_commit=current_git_commit(),
        extra={
            "dataset": str(args.dataset),
            "split": args.split,
            "variants": list(variants),
            "limit": args.limit,
            "concurrency": args.concurrency,
            "max_tokens": cfg.max_tokens,
            "timeout_s": cfg.timeout_s,
            "temperature": cfg.temperature,
            "input": "page images only (no text layer)",
            "schema": "PurchaseOrder.model_json_schema() (full names, computed fields, objects)",
        },
    )
    run_id = args.run_id or new_run_id("baseline")
    writer = RunWriter.create(args.runs_dir, run_id, config, overwrite=args.overwrite)
    print(f"Baseline B0: {len(docs)} documents -> {writer.run_dir} (model {cfg.model_alias})")

    start = time.perf_counter()
    records = asyncio.run(run_baseline(docs, cfg, writer, args.concurrency))
    ok = sum(r.status == "completed" for r in records)
    print(f"Done in {time.perf_counter() - start:.0f}s: {ok}/{len(records)} completed")


if __name__ == "__main__":
    main()
