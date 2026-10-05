"""Structured JSON logging.

Usage:
    configure_logging("INFO")
    log = get_logger(__name__)
    log.info("job completed", extra={"job_id": job_id, "latency_ms": 812})

Each record is written to stdout as one JSON object per line. Keys passed via `extra`
become top-level fields.
"""

import logging
import sys
from datetime import UTC, datetime
from typing import Any

import orjson

# Attributes every LogRecord has; anything else on a record came from `extra=`.
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None)).keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise `record`, including any `extra` fields, to a JSON string."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # default=str covers Decimal, Path, UUID and other non-JSON-native values.
        return orjson.dumps(payload, default=str).decode()


def configure_logging(level: str = "INFO") -> None:
    """Route all logging to stdout as JSON at `level`. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin wrapper so call sites import from one place)."""
    return logging.getLogger(name)
