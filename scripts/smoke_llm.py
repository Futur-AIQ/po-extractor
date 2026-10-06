"""Smoke-test one structured JSON call through LiteLLM using the `po-fast` alias.

Run from the repo root (needs `make model` and `make litellm` running):
    uv run python -m scripts.smoke_llm

Checks the request shape every extraction call will use: JSON-schema constrained output,
temperature 0, and thinking disabled via chat_template_kwargs. (With the openai client this
field would go in `extra_body`; with plain httpx it is simply a top-level request field.)
"""

import json
import sys
import time

import httpx

from app.core.settings import get_settings

SAMPLE_TEXT = """\
PURCHASE ORDER
PO No.: PO/2026/0417            Date: 14-03-2026
Buyer: Shree Ganesh Engineering Works, Pune
Vendor: Acme Bearings Pvt Ltd, Chennai
"""

PO_HEADER_SCHEMA = {
    "type": "object",
    "properties": {
        "po_number": {"type": "string"},
        "po_date": {"type": "string"},
    },
    "required": ["po_number", "po_date"],
    "additionalProperties": False,
}


def main() -> int:
    """Send the smoke request, print result, latency and token usage; return an exit code."""
    settings = get_settings()
    url = f"{settings.litellm_base_url.rstrip('/')}/v1/chat/completions"
    api_key = settings.litellm_api_key.get_secret_value()
    if not api_key:
        print("LITELLM_API_KEY is not set in .env (see .env.example)", file=sys.stderr)
        return 1
    headers = {"Authorization": f"Bearer {api_key}"}

    body = {
        "model": settings.po_fast,
        "temperature": 0,
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "Extract the PO number and PO date. Reply with JSON."},
            {"role": "user", "content": SAMPLE_TEXT},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "po_header", "strict": True, "schema": PO_HEADER_SCHEMA},
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }

    start = time.perf_counter()
    try:
        response = httpx.post(url, json=body, headers=headers, timeout=180)
    except httpx.HTTPError as exc:
        print(f"Request to {url} failed: {exc!r}. Is `make litellm` running?", file=sys.stderr)
        return 1
    latency_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        print(f"HTTP {response.status_code}: {response.text[:500]}", file=sys.stderr)
        return 1

    data = response.json()
    message = data["choices"][0]["message"]
    try:
        parsed = json.loads(message["content"])
    except (TypeError, json.JSONDecodeError):
        print(f"Response is not valid JSON: {message.get('content')!r}", file=sys.stderr)
        return 1
    if set(parsed) != {"po_number", "po_date"}:
        print(f"Unexpected keys in response: {sorted(parsed)}", file=sys.stderr)
        return 1

    usage = data.get("usage", {})
    print(f"model alias:        {settings.po_fast}")
    print(f"parsed JSON:        {json.dumps(parsed)}")
    print(f"latency:            {latency_ms:.0f} ms")
    print(f"prompt tokens:      {usage.get('prompt_tokens')}")
    print(f"completion tokens:  {usage.get('completion_tokens')}")
    if message.get("reasoning_content"):
        print("ERROR: model returned reasoning_content; thinking was NOT disabled", file=sys.stderr)
        return 1
    print("thinking:           off (no reasoning_content)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
