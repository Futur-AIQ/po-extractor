"""Thin Langfuse helper: a client built from settings plus trace/span context managers.

Usage:
    lf = get_langfuse()
    with trace(lf, "extract-po", metadata={"po_id": "PO-1"}):
        with span(lf, "pdf_text"):
            ...
        with span(lf, "llm_group", metadata={"group": "header"}):
            ...
    lf.flush()

If the Langfuse keys are not set, tracing is disabled and the context managers become
no-ops, so the pipeline runs the same with or without Langfuse.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from langfuse import Langfuse, LangfuseSpan, propagate_attributes

from app.core.settings import Settings, get_settings


def create_langfuse(settings: Settings) -> Langfuse:
    """Create a Langfuse client for the self-hosted instance described by `settings`."""
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key.get_secret_value()
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=settings.langfuse_host,
        tracing_enabled=bool(public_key and secret_key),
    )


@lru_cache
def get_langfuse() -> Langfuse:
    """Return the process-wide Langfuse client (created once from app settings)."""
    return create_langfuse(get_settings())


@contextmanager
def trace(
    client: Langfuse,
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[LangfuseSpan]:
    """Open a new trace with a root span; spans opened inside it become its children.

    `metadata` and `tags` are set at trace level (Langfuse stores trace metadata values as
    strings) and also on the root span with their original types.
    """
    trace_metadata = {k: str(v) for k, v in (metadata or {}).items()}
    with (
        propagate_attributes(trace_name=name, metadata=trace_metadata, tags=tags),
        client.start_as_current_observation(name=name, input=input, metadata=metadata) as root,
    ):
        yield root


@contextmanager
def span(
    client: Langfuse,
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[LangfuseSpan]:
    """Open a span nested under the currently active span (e.g. one pipeline stage)."""
    with client.start_as_current_observation(name=name, input=input, metadata=metadata) as s:
        yield s
