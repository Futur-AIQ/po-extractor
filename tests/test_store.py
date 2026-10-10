"""Tests for the SQLite service store (Step 4.1), on a temp database."""

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from app.store import repository as repo
from app.store.db import Database, init_db
from app.store.models import Job, utcnow


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "store.db")
    await init_db(database)
    yield database
    await database.dispose()


def sha(n: int) -> str:
    return f"{n:064x}"


def envelope(status: str = "completed", **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "error": None,
        "routing": {"strategy": "two_call", "retried_calls": []},
        "pages": [{"page": 1, "kind": "native"}, {"page": 2, "kind": "native"}],
        **extra,
    }


async def finish(db: Database, job_id: str, env: dict[str, Any]) -> Job:
    await repo.mark_processing(db, job_id)
    return await repo.save_result(db, job_id, env)


# --- Database settings -------------------------------------------------------------------


async def test_wal_mode_busy_timeout_and_foreign_keys(db: Database) -> None:
    async with db.engine.connect() as connection:
        pragma = connection.exec_driver_sql
        assert (await pragma("PRAGMA journal_mode")).scalar() == "wal"
        assert (await pragma("PRAGMA busy_timeout")).scalar() == 10_000
        assert (await pragma("PRAGMA foreign_keys")).scalar() == 1


async def test_init_db_is_idempotent(db: Database) -> None:
    await repo.create_job_if_absent(db, sha(1), "a.pdf", 1)
    await init_db(db)  # tables exist: nothing changes
    assert len(await repo.list_jobs(db)) == 1


# --- Jobs ----------------------------------------------------------------------------------


async def test_create_job_is_idempotent(db: Database) -> None:
    job, created = await repo.create_job_if_absent(db, sha(1), "po.pdf", 7, "http://cb.local/x")
    assert created and job.status == "queued" and job.po_seq == 7 and job.attempts == 0
    assert job.callback_url == "http://cb.local/x" and job.created_at.tzinfo is not None
    again, created = await repo.create_job_if_absent(db, sha(1), "renamed.pdf", 8)
    assert not created
    assert (again.filename, again.po_seq, again.created_at) == ("po.pdf", 7, job.created_at)
    assert len(await repo.list_jobs(db)) == 1


async def test_status_transitions_and_timings(db: Database) -> None:
    await repo.create_job_if_absent(db, sha(1), "po.pdf", 1)
    job = await repo.mark_processing(db, sha(1))
    assert job.status == "processing" and job.attempts == 1 and job.queue_ms >= 0
    first_queue = job.queue_ms
    job = await repo.mark_processing(db, sha(1))  # a retried worker attempt
    assert job.attempts == 2 and job.queue_ms == first_queue

    routing = {"strategy": "per_page", "retried_calls": ["LI-p2"]}
    pages = [{"page": 1, "kind": "native"}, {"page": 2, "kind": "scanned"}]
    job = await repo.save_result(db, sha(1), envelope(routing=routing, pages=pages))
    assert job.status == "completed" and job.finished_at >= job.started_at
    assert job.latency_ms >= job.queue_ms
    assert (job.strategy, job.retried, job.page_kinds) == ("per_page", True, ["native", "scanned"])

    with pytest.raises(repo.InvalidTransition):
        await repo.mark_processing(db, sha(1))  # finished jobs do not restart
    with pytest.raises(repo.InvalidTransition):
        await repo.mark_failed(db, sha(1), "late failure")


async def test_invalid_transitions_and_unknown_jobs(db: Database) -> None:
    await repo.create_job_if_absent(db, sha(1), "po.pdf", 1)
    with pytest.raises(repo.InvalidTransition):
        await repo.save_result(db, sha(1), envelope())  # never started
    await repo.mark_processing(db, sha(1))
    with pytest.raises(ValueError, match="not a final status"):
        await repo.save_result(db, sha(1), envelope(status="processing"))
    with pytest.raises(repo.JobNotFound):
        await repo.mark_processing(db, sha(99))
    assert await repo.get_job_with_result(db, sha(99)) is None


async def test_mark_failed_from_queued_and_processing(db: Database) -> None:
    for n in (1, 2):
        await repo.create_job_if_absent(db, sha(n), "po.pdf", n)
    await repo.mark_processing(db, sha(2))
    for n in (1, 2):
        job = await repo.mark_failed(db, sha(n), "retries exhausted (DLQ)")
        assert job.status == "failed" and job.error == "retries exhausted (DLQ)"
        assert job.latency_ms is not None and job.finished_at is not None


async def test_failed_envelope_is_stored_with_its_error(db: Database) -> None:
    await repo.create_job_if_absent(db, sha(1), "po.pdf", 1)
    job = await finish(db, sha(1), envelope(status="failed", error="LI: HTTP 503"))
    assert job.status == "failed" and job.error == "LI: HTTP 503"


async def test_get_job_with_result(db: Database) -> None:
    await repo.create_job_if_absent(db, sha(1), "po.pdf", 1)
    job, result = await repo.get_job_with_result(db, sha(1))
    assert job.status == "queued" and result is None
    await finish(db, sha(1), envelope(data={"po_number": "PO/1"}))
    job, result = await repo.get_job_with_result(db, sha(1))
    assert job.status == "completed" and result.envelope["data"] == {"po_number": "PO/1"}


async def test_list_jobs_filters(db: Database) -> None:
    for n in range(1, 6):
        await repo.create_job_if_absent(db, sha(n), f"{n}.pdf", n)
    await finish(db, sha(2), envelope())
    await finish(db, sha(4), envelope())
    jobs = await repo.list_jobs(db)
    assert [job.po_seq for job in jobs] == [5, 4, 3, 2, 1]  # newest first
    assert [job.po_seq for job in await repo.list_jobs(db, status="completed")] == [4, 2]
    assert [job.po_seq for job in await repo.list_jobs(db, limit=2)] == [5, 4]
    assert await repo.list_jobs(db, since=utcnow() + timedelta(seconds=1)) == []
    assert len(await repo.list_jobs(db, since=utcnow() - timedelta(minutes=1))) == 5
    with pytest.raises(ValueError):
        await repo.list_jobs(db, status="done")


# --- Events and reviews --------------------------------------------------------------------


async def test_events_in_order_and_after_id(db: Database) -> None:
    await repo.create_job_if_absent(db, sha(1), "po.pdf", 1)
    first = await repo.add_event(db, sha(1), "stage", {"stage": "preprocess"})
    await repo.add_event(db, sha(1), "stage", {"stage": "llm"})
    await repo.add_event(db, sha(1), "done")
    events = await repo.list_events(db, sha(1))
    assert [e.payload for e in events] == [{"stage": "preprocess"}, {"stage": "llm"}, None]
    assert [e.type for e in await repo.list_events(db, sha(1), after_id=first.id)] == [
        "stage",
        "done",
    ]
    assert await repo.list_events(db, sha(2)) == []


async def test_needs_review_opens_a_review_and_resolve_closes_it(db: Database) -> None:
    await repo.create_job_if_absent(db, sha(1), "po.pdf", 1)
    await repo.create_job_if_absent(db, sha(2), "ok.pdf", 2)
    review_info = {"failed_rules": ["gstin"], "failed_fields": ["vendor_gstin"]}
    job = await finish(db, sha(1), envelope(status="needs_review", review=review_info))
    await finish(db, sha(2), envelope())
    assert job.status == "needs_review"

    [review] = await repo.list_review(db)
    assert review.job_id == sha(1) and not review.resolved
    assert (review.failed_rules, review.failed_fields) == (["gstin"], ["vendor_gstin"])

    resolved = await repo.resolve_review(db, sha(1), {"vendor_gstin": "27AAPFU0939F1ZV"})
    assert resolved.resolved and resolved.resolved_at is not None
    assert resolved.corrections == {"vendor_gstin": "27AAPFU0939F1ZV"}
    assert await repo.list_review(db) == []
    assert len(await repo.list_review(db, open_only=False)) == 1
    with pytest.raises(repo.JobNotFound):
        await repo.resolve_review(db, sha(2), {})  # a completed job has no review


# --- Metrics -------------------------------------------------------------------------------


async def test_metrics_percentiles_on_known_values(db: Database) -> None:
    latencies = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    kinds = [["native"], ["native", "native"], ["scanned"], ["native", "scanned"], None]
    statuses = ["completed", "completed", "needs_review", "needs_review", "failed"]
    for n, (latency, page_kinds, status) in enumerate(
        zip(latencies, kinds, statuses, strict=True), 1
    ):
        await repo.create_job_if_absent(db, sha(n), f"{n}.pdf", n)
        async with db.sessions.begin() as session:
            job = await session.get(Job, sha(n))
            job.status, job.latency_ms, job.queue_ms = status, latency, latency / 10
            job.page_kinds, job.retried = page_kinds, n in (1, 3)
    await repo.create_job_if_absent(db, sha(6), "queued.pdf", 6)  # no latency yet
    await repo.create_job_if_absent(db, sha(7), "old.pdf", 0)
    async with db.sessions.begin() as session:  # outside the window
        old = await session.get(Job, sha(7))
        old.created_at, old.status, old.latency_ms = utcnow() - timedelta(hours=3), "completed", 9e9

    m = await repo.metrics_summary(db, window_minutes=60)
    assert m.jobs == 6
    assert m.by_status == {
        "queued": 1,
        "processing": 0,
        "completed": 2,
        "needs_review": 2,
        "failed": 1,
    }
    assert m.latency_p50_ms == 3000.0
    assert m.latency_p95_ms == pytest.approx(4800.0)  # linear interpolation: 4000 + 0.8 x 1000
    assert m.latency_max_ms == 5000.0
    assert m.queue_p95_ms == pytest.approx(480.0)
    assert m.page_mix == {"native": 2, "scanned": 1, "mixed": 1}
    assert m.retry_rate == pytest.approx(0.5)  # 2 of the 4 completed / needs_review jobs


async def test_metrics_on_an_empty_store(db: Database) -> None:
    m = await repo.metrics_summary(db)
    assert m.jobs == 0 and m.latency_p95_ms is None and m.retry_rate is None


# --- Concurrency ---------------------------------------------------------------------------


async def test_concurrent_writes_from_two_processes_do_not_lock(tmp_path: Path) -> None:
    """Two engines on one file (like the API and the worker) writing at the same time.

    Every job is created by one and processed by the other, so both read-then-write the same
    rows concurrently: the case where a deferred BEGIN fails with "database is locked".
    """
    path = tmp_path / "shared.db"
    api, worker = Database(path, busy_timeout_ms=5000), Database(path, busy_timeout_ms=5000)
    await init_db(api)

    count = 40
    submitted = [asyncio.Event() for _ in range(count)]

    async def submit(n: int) -> None:
        await repo.create_job_if_absent(api, sha(n), f"{n}.pdf", n)
        submitted[n].set()
        await repo.add_event(api, sha(n), "queued")

    async def process(n: int) -> None:
        await submitted[n].wait()
        await repo.mark_processing(worker, sha(n))
        await repo.add_event(worker, sha(n), "stage", {"stage": "llm"})
        await repo.save_result(worker, sha(n), envelope())

    await asyncio.gather(*(submit(n) for n in range(count)), *(process(n) for n in range(count)))
    jobs = await repo.list_jobs(api, limit=1000)
    assert len(jobs) == count and {job.status for job in jobs} == {"completed"}
    assert sum([len(await repo.list_events(api, sha(n))) for n in range(count)]) == 2 * count
    await api.dispose()
    await worker.dispose()
