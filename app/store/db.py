"""SQLite database for the service: async engine (aiosqlite), WAL mode, busy timeout.

The API and the worker are separate processes writing the same file:
- WAL mode: readers never block the writer and the writer never blocks readers.
- busy_timeout: a second writer waits for the lock instead of failing with
  "database is locked".
- Every transaction starts with BEGIN IMMEDIATE, taking the write lock up front. With SQLite's
  default (deferred) BEGIN, two sessions that both read and then write can deadlock, and SQLite
  fails one of them at once without waiting. Transactions here are short, so taking the lock
  early costs little.
"""

from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.core.settings import Settings
from app.store.models import Base


class Database:
    """The engine and session factory for one SQLite file."""

    def __init__(self, path: Path, busy_timeout_ms: int = 10_000) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.engine: AsyncEngine = create_async_engine(
            f"sqlite+aiosqlite:///{path}",
            connect_args={"timeout": busy_timeout_ms / 1000},  # sqlite3's own wait for locks
        )

        @event.listens_for(self.engine.sync_engine, "connect")
        def _configure(dbapi_connection: Any, _record: Any) -> None:
            dbapi_connection.isolation_level = None  # we issue BEGIN ourselves (below)
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            cursor.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, fewer fsyncs
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        @event.listens_for(self.engine.sync_engine, "begin")
        def _begin_immediate(connection: Any) -> None:
            connection.exec_driver_sql("BEGIN IMMEDIATE")

        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Database":
        return cls(settings.sqlite_path, settings.sqlite_busy_timeout_ms)

    async def dispose(self) -> None:
        await self.engine.dispose()


async def init_db(db: Database) -> None:
    """Create the tables if they do not exist."""
    async with db.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
