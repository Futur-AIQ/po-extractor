"""Tables of the service store (SQLite): jobs, results, events and the review queue.

All timestamps are UTC. SQLite keeps no time zone, so UtcDateTime stores naive UTC and gives
back timezone-aware datetimes.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JOB_STATUSES = ("queued", "processing", "completed", "needs_review", "failed")
FINAL_STATUSES = ("completed", "needs_review", "failed")


def utcnow() -> datetime:
    return datetime.now(UTC)


class UtcDateTime(TypeDecorator):
    """A datetime stored as naive UTC and read back as timezone-aware UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None


class Base(DeclarativeBase):
    pass


class Job(Base):
    """One uploaded PO. The id is the file's SHA-256, so a duplicate upload is the same job."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), index=True, default="queued")
    po_seq: Mapped[int] = mapped_column(Integer, index=True)  # arrival order = LLM priority
    attempts: Mapped[int] = mapped_column(Integer, default=0)  # times processing started
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, index=True, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    error: Mapped[str | None] = mapped_column(Text)
    callback_url: Mapped[str | None] = mapped_column(String(2048))
    trace_url: Mapped[str | None] = mapped_column(String(2048))
    page_kinds: Mapped[list[str] | None] = mapped_column(JSON)  # per page: native / scanned
    strategy: Mapped[str | None] = mapped_column(String(16))  # two_call / per_page
    retried: Mapped[bool] = mapped_column(default=False)  # a po-accurate retry ran
    latency_ms: Mapped[float | None]  # upload accepted -> result stored (PRD §3)
    queue_ms: Mapped[float | None]  # upload accepted -> processing started


class Result(Base):
    """The result envelope of a finished job (PRD §6.4)."""

    __tablename__ = "results"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    envelope: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class Event(Base):
    """A progress event of a job (stage started/finished, call done, ...), for live updates."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class Review(Base):
    """A job that needs a human: the failed rules and fields, and the reviewer's corrections."""

    __tablename__ = "reviews"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    failed_rules: Mapped[list[str]] = mapped_column(JSON)
    failed_fields: Mapped[list[str]] = mapped_column(JSON)
    resolved: Mapped[bool] = mapped_column(default=False, index=True)
    corrections: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
