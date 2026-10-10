"""Run the optimised pipeline over a dataset split and write the standard run format.

    uv run python -m app.extraction.run --split dev --variants N,S,M --strategy adaptive \\
        --run-id opt_dev_local --concurrency 4

Documents are submitted concurrently, at most --concurrency at a time, each with an
increasing po_priority (its submission order), as a burst would arrive. The process-wide
LLM limit (LLM_CONCURRENCY) still caps the calls in flight across documents.

Ablations: --strategy, --dpi, --native-mode, --no-skip-pages, --no-retry. Every choice is
recorded in config.json; each document's routing (strategy, reason, retried calls) is in
records.jsonl, and the raw answer of every call is in raw/<doc_id>.json.

Needs the local model and LiteLLM (`PARALLEL=4 CTX_SIZE=65536 make model`, `make litellm`).
"""

import argparse
import asyncio
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app.common.stats import percentile
from app.core.settings import Settings, get_settings
from app.extraction.llm_client import LlmClient
from app.extraction.orchestrator import LlmCaller
from app.extraction.pipeline import ResultEnvelope, extract
from app.extraction.prompts import PROMPT_VERSION
from app.extraction.validate import Masters, load_masters
from eval.dataset import VARIANTS, DatasetDoc, select_docs
from eval.run_format import (
    DocRecord,
    RunConfig,
    RunWriter,
    TokenUsage,
    current_git_commit,
    new_run_id,
)


def token_usage(env: ResultEnvelope) -> TokenUsage:
    """Tokens summed over all calls of a document (retries included)."""
    calls = env.tokens.values()
    reasoning = [c["reasoning"] for c in calls if c.get("reasoning") is not None]
    return TokenUsage(
        prompt=sum(c["in"] or 0 for c in calls),
        completion=sum(c["out"] or 0 for c in calls),
        reasoning=sum(reasoning) if reasoning else None,
    )


def save_result(
    writer: RunWriter, doc: DatasetDoc, env: ResultEnvelope, raw_calls: list[dict[str, Any]]
) -> DocRecord:
    """Write the output (or error record), raw calls and record line of one document."""
    if env.data is not None:
        writer.write_output(doc.doc_id, env.data, env.status, env.error)
    else:
        writer.write_error(doc.doc_id, env.status, env.error or "unknown error")
    envelope = env.model_dump(mode="json", exclude={"data"})
    writer.write_raw(doc.doc_id, {"calls": raw_calls, "envelope": envelope})
    record = DocRecord(
        doc_id=doc.doc_id,
        variant=doc.variant,
        status=env.status,
        timings_ms=env.timings_ms,
        tokens=token_usage(env),
        calls=len(env.timeline),
        error=env.error or _review_text(env),
        routing=env.routing.model_dump() if env.routing else None,
    )
    writer.append_record(record)
    return record


def _review_text(env: ResultEnvelope) -> str | None:
    if env.review is None:
        return None
    return f"review: {', '.join(env.review.failed_rules)}"


def progress_line(index: int, total: int, record: DocRecord) -> str:
    routing = record.routing or {}
    retried = routing.get("retried_calls") or []
    route = f"{routing.get('strategy', '-')}{' retry=' + '+'.join(retried) if retried else ''}"
    error = f"  {record.error[:120]}" if record.error else ""
    return (
        f"[{index}/{total}] {record.doc_id:<10} {record.status:<12} {record.total_ms / 1000:6.1f}s "
        f"{route:<24} calls={record.calls} completion={record.tokens.completion}{error}"
    )


async def run_extraction(
    docs: Sequence[DatasetDoc],
    settings: Settings,
    writer: RunWriter,
    client: LlmCaller,
    concurrency: int = 1,
    masters: Masters | None = None,
    progress: Callable[[str], None] = print,
) -> list[DocRecord]:
    """Extract every document, at most `concurrency` at a time; returns the records."""
    semaphore = asyncio.Semaphore(concurrency)
    done: list[DocRecord] = []

    async def one(priority: int, doc: DatasetDoc) -> None:
        queued = time.perf_counter()
        async with semaphore:
            queue_ms = round((time.perf_counter() - queued) * 1000, 1)
            pdf = await asyncio.to_thread(doc.pdf.read_bytes)
            raw: list[dict[str, Any]] = []
            env = await extract(
                pdf,
                client,
                settings,
                job_id=doc.doc_id,
                po_priority=priority,
                masters=masters,
                queue_ms=queue_ms,
                raw_calls=raw,
            )
            record = await asyncio.to_thread(save_result, writer, doc, env, raw)
            done.append(record)
            progress(progress_line(len(done), len(docs), record))

    await asyncio.gather(*(one(priority, doc) for priority, doc in enumerate(docs, start=1)))
    return done


def summary(records: list[DocRecord], seconds: float) -> str:
    """One-paragraph result of the run, for the terminal."""
    counts = {
        s: sum(r.status == s for r in records) for s in ("completed", "needs_review", "failed")
    }
    totals = [r.total_ms / 1000 for r in records]
    retried = sum(bool((r.routing or {}).get("retried_calls")) for r in records)
    completion = sum(r.tokens.completion for r in records) / max(len(records), 1)
    p50, p95 = percentile(totals, 50), percentile(totals, 95)
    return (
        f"Done in {seconds:.0f}s: {counts['completed']} completed, {counts['needs_review']} "
        f"needs_review, {counts['failed']} failed; {retried} retried. Per PO: p50 "
        f"{p50 or 0:.1f}s, p95 {p95 or 0:.1f}s, mean completion tokens {completion:.0f}"
    )


def parse_args(argv: Sequence[str] | None, settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the optimised extraction pipeline.")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--variants", default="N,S,M", help=f"subset of {','.join(VARIANTS)}")
    parser.add_argument("--limit", type=int, default=None, help="number of POs (default: all)")
    parser.add_argument("--concurrency", type=int, default=4, help="documents in flight")
    parser.add_argument("--run-id", default=None, help="default: optimised_<timestamp>")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run")
    parser.add_argument(
        "--strategy", choices=["two_call", "per_page", "adaptive"], default=settings.call_strategy
    )
    parser.add_argument("--dpi", type=int, default=settings.image_dpi)
    parser.add_argument(
        "--native-mode", choices=["text+image", "text", "image"], default=settings.native_mode
    )
    parser.add_argument("--no-skip-pages", action="store_true", help="send field-less pages too")
    parser.add_argument("--no-retry", action="store_true", help="no po-accurate retry")
    parser.add_argument("--dataset", type=Path, default=Path(settings.data_dir) / "synthetic")
    parser.add_argument("--runs-dir", type=Path, default=Path(settings.runs_dir))
    return parser.parse_args(argv)


def run_settings(args: argparse.Namespace, settings: Settings) -> Settings:
    """Settings with the CLI's ablation choices applied."""
    return settings.model_copy(
        update={
            "call_strategy": args.strategy,
            "image_dpi": args.dpi,
            "native_mode": args.native_mode,
            "skip_fieldless_pages": not args.no_skip_pages,
            "max_reasks": 0 if args.no_retry else settings.max_reasks,
        }
    )


def run_config(args: argparse.Namespace, s: Settings, variants: tuple[str, ...]) -> RunConfig:
    return RunConfig(
        pipeline="optimised",
        model_alias=s.po_fast,
        strategy=s.call_strategy,
        dpi=s.image_dpi,
        thinking="off (retries: thinking on)" if s.max_reasks else "off",
        git_commit=current_git_commit(),
        extra={
            "dataset": str(args.dataset),
            "split": args.split,
            "variants": list(variants),
            "limit": args.limit,
            "concurrency": args.concurrency,
            "llm_concurrency": s.llm_concurrency,
            "native_mode": s.native_mode,
            "text_engine": s.text_engine,
            "image_format": s.image_format,
            "skip_fieldless_pages": s.skip_fieldless_pages,
            "max_reasks": s.max_reasks,
            "retry_alias": s.po_accurate,
            "retry_deadline_s": s.retry_deadline_s,
            "llm_profile_set": s.llm_profile_set,
            "llm_timeout_s": s.llm_timeout_s,
            "max_tokens": {"header": s.llm_max_tokens_header, "lines": s.llm_max_tokens_lines},
            "adaptive_thresholds": {
                "rows": s.adaptive_row_threshold,
                "pages": s.adaptive_page_threshold,
            },
            "prompt_version": PROMPT_VERSION,
        },
    )


def main(argv: Sequence[str] | None = None, client: LlmCaller | None = None) -> list[DocRecord]:
    """CLI entry point. Tests pass `argv` and a fake `client`."""
    base = get_settings()
    args = parse_args(argv, base)
    if client is None and not base.litellm_api_key.get_secret_value():
        sys.exit("LITELLM_API_KEY is not set in .env (see .env.example)")
    split = None if args.split == "all" else args.split
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    docs = select_docs(args.dataset, split, variants, args.limit)
    if not docs:
        sys.exit(f"no documents for split={args.split} variants={args.variants}")
    settings = run_settings(args, base)
    masters_dir = args.dataset / "masters"
    masters = load_masters(masters_dir) if (masters_dir / "parties.json").exists() else None

    writer = RunWriter.create(
        args.runs_dir,
        args.run_id or new_run_id("optimised"),
        run_config(args, settings, variants),
        overwrite=args.overwrite,
    )
    print(
        f"Optimised pipeline: {len(docs)} documents -> {writer.run_dir} "
        f"(strategy {settings.call_strategy}, model {settings.po_fast})"
    )

    async def go() -> list[DocRecord]:
        if client is not None:
            return await run_extraction(docs, settings, writer, client, args.concurrency, masters)
        async with LlmClient(settings) as llm:
            return await run_extraction(docs, settings, writer, llm, args.concurrency, masters)

    start = time.perf_counter()
    records = asyncio.run(go())
    print(summary(records, time.perf_counter() - start))
    return records


if __name__ == "__main__":
    main()
