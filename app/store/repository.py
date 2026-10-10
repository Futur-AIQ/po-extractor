"""Repository: every read and write of the service store, one short transaction each.

Job lifecycle:

    queued --mark_processing--> processing --save_result--> completed | needs_review
       |                           |  ^ (a retried worker attempt starts again)
       +-----------mark_failed-----+--+-------------------> failed

Latency is end to end (PRD §3): from the upload being accepted (created_at) to the result
being stored (finished_at). Queue time is from created_at to the first start of processing.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from app.common.stats import percentile
from app.store.db import Database
from app.store.models import JOB_STATUSES, Event, Job, Result, Review, utcnow


class InvalidTransition(ValueError):
    """A job status change that the lifecycle does not allow."""


class JobNotFound(KeyError):
    """No job with that id."""


def _ms_between(start: datetime, end: datetime) -> float:
    return round((end - start).total_seconds() * 1000, 1)


def _require(job: Job | None, job_id: str, allowed: tuple[str, ...], action: str) -> Job:
    if job is None:
        raise JobNotFound(job_id)
    if job.status not in allowed:
        raise InvalidTransition(f"cannot {action} job {job_id[:12]} in status {job.status}")
    return job


# =========================================================================================
# Jobs
# =========================================================================================


async def create_job_if_absent(
    db: Database, job_id: str, filename: str, po_seq: int, callback_url: str | None = None
) -> tuple[Job, bool]:
    """(job, created). A job with this id (the file's SHA-256) is returned as it is."""
    async with db.sessions.begin() as session:
        values = {
            "id": job_id,
            "filename": filename,
            "po_seq": po_seq,
            "callback_url": callback_url,
            "status": "queued",
            "attempts": 0,
            "created_at": utcnow(),
            "retried": False,
        }
        statement = insert(Job).values(**values).on_conflict_do_nothing(index_elements=["id"])
        created = (await session.execute(statement)).rowcount == 1
        job = await session.get(Job, job_id)
        assert job is not None
        return job, created


async def mark_processing(db: Database, job_id: str) -> Job:
    """A worker starts (or restarts) the job: attempts + 1; queue time on the first start."""
    async with db.sessions.begin() as session:
        job = _require(await session.get(Job, job_id), job_id, ("queued", "processing"), "start")
        now = utcnow()
        if job.queue_ms is None:
            job.queue_ms = _ms_between(job.created_at, now)
        job.status, job.started_at, job.attempts = "processing", now, job.attempts + 1
        return job


async def save_result(db: Database, job_id: str, envelope: Mapping[str, Any]) -> Job:
    """Store the result envelope (JSON) and finish the job with the envelope's status.

    A needs_review result opens a Review with the failed rules and fields.
    """
    status = envelope["status"]
    if status not in ("completed", "needs_review", "failed"):
        raise ValueError(f"envelope status {status!r} is not a final status")
    routing = envelope.get("routing") or {}
    async with db.sessions.begin() as session:
        job = _require(await session.get(Job, job_id), job_id, ("processing",), "save a result for")
        now = utcnow()
        job.status, job.finished_at = status, now
        job.latency_ms = _ms_between(job.created_at, now)
        job.error = envelope.get("error")
        job.strategy = routing.get("strategy")
        job.retried = bool(routing.get("retried_calls"))
        job.page_kinds = [page["kind"] for page in envelope.get("pages", [])] or None
        await session.merge(Result(job_id=job_id, envelope=dict(envelope), created_at=now))

        if status == "needs_review":
            review = await session.get(Review, job_id)
            details = envelope.get("review") or {}
            if review is None:
                review = Review(job_id=job_id, failed_rules=[], failed_fields=[], created_at=now)
                session.add(review)
            review.failed_rules = list(details.get("failed_rules", []))
            review.failed_fields = list(details.get("failed_fields", []))
            review.resolved, review.corrections, review.resolved_at = False, None, None
        return job


async def mark_failed(db: Database, job_id: str, error: str) -> Job:
    """The job failed outside the pipeline's own result (e.g. retries exhausted, DLQ)."""
    async with db.sessions.begin() as session:
        job = _require(await session.get(Job, job_id), job_id, ("queued", "processing"), "fail")
        now = utcnow()
        job.status, job.finished_at, job.error = "failed", now, error
        job.latency_ms = _ms_between(job.created_at, now)
        return job


async def list_jobs(
    db: Database, status: str | None = None, limit: int = 100, since: datetime | None = None
) -> list[Job]:
    """Newest jobs first, optionally only one status and/or created at or after `since`."""
    if status is not None and status not in JOB_STATUSES:
        raise ValueError(f"unknown status {status!r}")
    query = select(Job).order_by(Job.created_at.desc(), Job.po_seq.desc()).limit(limit)
    if status is not None:
        query = query.where(Job.status == status)
    if since is not None:
        query = query.where(Job.created_at >= since)
    async with db.sessions.begin() as session:
        return list((await session.scalars(query)).all())


async def get_job_with_result(db: Database, job_id: str) -> tuple[Job, Result | None] | None:
    """The job and its result (None while it runs), or None if there is no such job."""
    async with db.sessions.begin() as session:
        job = await session.get(Job, job_id)
        if job is None:
            return None
        return job, await session.get(Result, job_id)


# =========================================================================================
# Events
# =========================================================================================


async def add_event(
    db: Database, job_id: str, type: str, payload: dict[str, Any] | None = None
) -> Event:
    """Record a progress event of a job."""
    async with db.sessions.begin() as session:
        event = Event(job_id=job_id, type=type, payload=payload, ts=utcnow())
        session.add(event)
        await session.flush()  # assigns the id
        return event


async def list_events(db: Database, job_id: str, after_id: int = 0) -> list[Event]:
    """Events of a job in order; `after_id` returns only newer ones (for live streams)."""
    query = select(Event).where(Event.job_id == job_id, Event.id > after_id).order_by(Event.id)
    async with db.sessions.begin() as session:
        return list((await session.scalars(query)).all())


# =========================================================================================
# Review queue
# =========================================================================================


async def list_review(db: Database, open_only: bool = True, limit: int = 100) -> list[Review]:
    """Jobs waiting for a reviewer (oldest first), or every review with open_only=False."""
    query = select(Review).order_by(Review.created_at).limit(limit)
    if open_only:
        query = query.where(Review.resolved.is_(False))
    async with db.sessions.begin() as session:
        return list((await session.scalars(query)).all())


async def resolve_review(db: Database, job_id: str, corrections: dict[str, Any]) -> Review:
    """Close a review with the reviewer's corrections (field -> corrected value)."""
    async with db.sessions.begin() as session:
        review = await session.get(Review, job_id)
        if review is None:
            raise JobNotFound(f"no review for job {job_id}")
        review.resolved, review.corrections, review.resolved_at = True, corrections, utcnow()
        return review


# =========================================================================================
# Metrics
# =========================================================================================


@dataclass(frozen=True)
class Metrics:
    """Service metrics over the jobs created in the last `window_minutes`."""

    window_minutes: int
    jobs: int
    by_status: dict[str, int]
    latency_p50_ms: float | None  # finished jobs, end to end
    latency_p95_ms: float | None
    latency_max_ms: float | None
    queue_p95_ms: float | None
    page_mix: dict[str, int] = field(default_factory=dict)  # native / scanned / mixed POs
    retry_rate: float | None = None  # share of validated jobs that needed a retry


def _page_mix(page_kinds: list[str]) -> str:
    kinds = set(page_kinds)
    return kinds.pop() if len(kinds) == 1 else "mixed"


async def metrics_summary(
    db: Database, window_minutes: int = 60, now: datetime | None = None
) -> Metrics:
    """Counts by status, latency p50/p95/max, queue p95, page mix and retry rate."""
    since = (now or utcnow()) - timedelta(minutes=window_minutes)
    async with db.sessions.begin() as session:
        jobs = list((await session.scalars(select(Job).where(Job.created_at >= since))).all())
    by_status = {status: sum(job.status == status for job in jobs) for status in JOB_STATUSES}
    latencies = [job.latency_ms for job in jobs if job.latency_ms is not None]
    queues = [job.queue_ms for job in jobs if job.queue_ms is not None]
    mix = {"native": 0, "scanned": 0, "mixed": 0}
    for job in jobs:
        if job.page_kinds:
            mix[_page_mix(job.page_kinds)] += 1
    validated = [job for job in jobs if job.status in ("completed", "needs_review")]
    return Metrics(
        window_minutes=window_minutes,
        jobs=len(jobs),
        by_status=by_status,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_max_ms=max(latencies, default=None),
        queue_p95_ms=percentile(queues, 95),
        page_mix=mix,
        retry_rate=sum(job.retried for job in validated) / len(validated) if validated else None,
    )
