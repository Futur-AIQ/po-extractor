"""Tests for the Langfuse tracing helper (Step 0.3). No network: spans go to memory."""

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.settings import Settings
from app.tracing.langfuse_client import create_langfuse, span, trace


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def client(exporter: InMemorySpanExporter) -> Langfuse:
    return Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )


def test_spans_nest_under_trace(client: Langfuse, exporter: InMemorySpanExporter) -> None:
    with trace(client, "extract-po", metadata={"pages": 2}, tags=["test"]) as root:
        with span(client, "pdf_text"):
            pass
        with span(client, "llm_group", metadata={"group": "header"}):
            pass
    client.flush()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"extract-po", "pdf_text", "llm_group"}
    root_span = spans["extract-po"]
    assert root_span.parent is None
    for child in ("pdf_text", "llm_group"):
        assert spans[child].parent.span_id == root_span.context.span_id
        assert spans[child].context.trace_id == root_span.context.trace_id
    assert root.trace_id == format(root_span.context.trace_id, "032x")
    assert spans["pdf_text"].attributes["langfuse.trace.name"] == "extract-po"
    assert spans["pdf_text"].attributes["langfuse.trace.tags"] == ("test",)


def test_tracing_disabled_without_keys() -> None:
    lf = create_langfuse(Settings(_env_file=None, langfuse_public_key=""))

    # Context managers must still work as no-ops so the pipeline runs without Langfuse.
    with trace(lf, "noop"), span(lf, "child"):
        pass
    lf.flush()
