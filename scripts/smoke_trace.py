"""Send one smoke-test trace (root span + two child spans) to the self-hosted Langfuse.

Run from the repo root:  uv run python -m scripts.smoke_trace
Needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env (see infra/langfuse/README.md).
"""

import sys
import time

from app.core.settings import get_settings
from app.tracing.langfuse_client import get_langfuse, span, trace


def main() -> int:
    """Send the smoke trace and print its URL; return a process exit code."""
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key.get_secret_value()):
        print("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set in .env", file=sys.stderr)
        return 1

    lf = get_langfuse()
    try:
        authenticated = lf.auth_check()
    except Exception as exc:  # SDK raises on 401 and on connection errors
        print(f"Langfuse auth check error: {exc.__class__.__name__}", file=sys.stderr)
        authenticated = False
    if not authenticated:
        print(f"Langfuse auth check failed against {settings.langfuse_host}", file=sys.stderr)
        return 1

    with trace(
        lf,
        "smoke-trace",
        input={"po_id": "SMOKE-0001"},
        metadata={"source": "scripts/smoke_trace.py", "pages": 2},
        tags=["smoke"],
    ) as root:
        with span(lf, "pdf_text", metadata={"pages": 2}) as s:
            time.sleep(0.05)
            s.update(output={"chars": 4210})
        with span(lf, "llm_group", metadata={"group": "header", "model": settings.po_fast}) as s:
            time.sleep(0.1)
            s.update(output={"fields_extracted": 12})
        root.update(output={"status": "completed"})
        trace_id = root.trace_id

    lf.flush()
    print(f"Sent trace {trace_id}")
    print(lf.get_trace_url(trace_id=trace_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
