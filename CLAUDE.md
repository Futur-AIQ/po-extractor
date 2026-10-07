# CLAUDE.md — po-extractor

Low-latency, high-accuracy purchase order (PO) extraction system. Interview demo built to production standards.
Full requirements: `docs/PRD.md` (v1.1). Build plan: `docs/PLAN.md`. Only implement the step you are asked to implement.

## Priorities (in order)
1. **Accuracy** — never trade correctness for speed without evidence. Never pass a PO that fails validation as valid.
2. **Latency** — p95 < 30 s end to end at 10 concurrent POs (burst of 20 < 60 s). Minimise LLM output tokens; maximise parallelism.
3. **Clarity** — simple, readable code an interviewer can follow. No clever abstractions.
4. **Confidentiality** — no external calls or telemetry from any component; PO data stays local.

## Tech stack
Python 3.12 + uv · FastAPI · Redis + arq · SQLite (SQLAlchemy async) · PyMuPDF + pymupdf4llm · Pydantic v2 · LiteLLM proxy · vLLM (H100) / llama.cpp (Mac dev) · Langfuse self-hosted · Jinja2 + Playwright + Faker(en_IN) · httpx · pandas + matplotlib · ruff + pytest.

## Repository layout
```
app/
  api/          FastAPI routes, SSE
  core/         settings (pydantic-settings), logging
  extraction/   pdf_text, regex_hints, prompts, groups, llm_client, orchestrator, merge, validate, routing
  worker/       arq worker, retries, dlq
  store/        SQLite models and repository
  tracing/      Langfuse helpers
  dashboard/    static index.html (vanilla JS, Chart.js, SSE)
schema/         Pydantic PO schema = single source of truth (JSON schema for LLM derives from it)
datagen/        synthetic PO generator, templates/, render, scanify
baseline/       naive pipeline (B0)
eval/           normalise, score, report
bench/          load generator, vLLM metrics, matrix runner, report
infra/          docker-compose.yml, litellm/, vllm/, langfuse/, lightning/
data/           generated data (gitignored)
docs/           PRD.md, PLAN.md, prompts/
tests/
```

## Conventions
- Async everywhere on I/O paths. No blocking calls inside the event loop (CPU work → `asyncio.to_thread` or process pool).
- All config via `app/core/settings.py` and `.env`; never hard-code URLs, ports, keys, or model names.
- The app calls **only LiteLLM** using model aliases `po-fast` (thinking off), `po-accurate` (same model, thinking low — used only for retries), `po-baseline`, and optional `po-moe`. Never call vLLM or llama.cpp directly from app code.
- Every LLM call: JSON-schema constrained, `temperature=0`, `presence_penalty=0`, `repetition_penalty=1.0`, `max_tokens` safety cap, thinking off via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` (except baseline and retries), records latency + token usage.
- LLM output contract: short keys (mapped back in code), absent fields omitted, line items as compact arrays, never request `COMPUTED_FIELDS` (computed in code).
- Pages: native → text layer + image; scanned → image only. Classification is per page.
- Message order is fixed for prefix caching: system → page content (per page: image, then text layer if native) → field definitions + task. Do not reorder.
- Money/quantities use `Decimal`, never float, in validation and scoring.
- Type hints everywhere; small functions; docstrings on public functions.
- Tests: pytest; mock the LLM in unit tests; no network in unit tests.
- Lint: `uv run ruff check . && uv run ruff format .`

## Commands
- `make up` / `make down` — app services (Redis)
- `make langfuse-up` / `make langfuse-down`
- `make model` — start local llama.cpp server; `make litellm` — start LiteLLM proxy
- `make doctor` — health check of all services
- `make test` · `make lint`
- (added in later phases) `make dataset`, `make eval`, `make bench`

## Ports
API 8080 · app Redis 6380 · LiteLLM 4000 · llama.cpp 8081 · Langfuse 3000 · vLLM via SSH tunnel 8000

## Definition of done for every step
Code + tests pass + ruff clean + acceptance criteria from `docs/PLAN.md` met + short summary of what changed and how to verify.
Do not start the next step. Do not refactor unrelated code.
