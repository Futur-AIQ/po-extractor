# Phase 0 — Foundations (Claude Code prompts)

Before starting: complete "Manual prerequisites" in `docs/PLAN.md`.
Run `/clear` before each prompt. Commit after each step.

---

## Step 0.1 — Repository scaffold

```
Read CLAUDE.md, docs/PRD.md (sections 8 and 10) and docs/PLAN.md (Phase 0).

Implement Step 0.1 only: the repository scaffold.

Tasks:
1. Initialise a uv project for Python 3.12 named po-extractor. Add runtime dependencies:
   fastapi, uvicorn[standard], pydantic>=2, pydantic-settings, httpx, arq, redis, sqlalchemy[asyncio],
   aiosqlite, pymupdf, pymupdf4llm, jinja2, faker, num2words, python-multipart, langfuse, orjson.
   Dev dependencies: pytest, pytest-asyncio, ruff, litellm[proxy], pandas, matplotlib, playwright.
2. Create the folder layout exactly as in CLAUDE.md, each Python package with an __init__.py.
3. app/core/settings.py using pydantic-settings, loading from .env, with fields for: API host/port,
   app Redis URL (port 6380), LiteLLM base URL and key, model aliases (po_fast, po_accurate, po_baseline),
   global LLM concurrency limit (default 32), max re-asks (default 1), Langfuse host/public/secret keys,
   SQLite path, data dir. Provide sensible defaults matching the ports in CLAUDE.md.
4. app/core/logging.py: structured JSON logging helper.
5. .env.example with every setting documented; .gitignore (include .env, data/, infra/langfuse/vendor/,
   __pycache__, .venv, results/, *.pdf under data).
6. ruff config in pyproject (line length 100, sensible rule set), pytest config (asyncio mode auto).
7. Makefile with targets: install, test, lint, format.
8. tests/test_settings.py verifying defaults load.

Do not implement any business logic. Do not create docker files yet.
Finish with: files created, and the exact commands I should run to verify.
```

**Verify:** `make install && make lint && make test`

---

## Step 0.2 — Docker Compose for app services

```
Read CLAUDE.md and docs/PLAN.md (Step 0.2).

Implement Step 0.2 only.

Tasks:
1. infra/docker-compose.yml with a single service: redis:7-alpine, container name po-redis,
   host port 6380 → container 6379, append-only persistence to a named volume, healthcheck.
2. Makefile targets: up, down, logs, ping (redis-cli -p 6380 ping via docker exec).
3. A short section in README.md (create it) called "Local services" explaining these targets.

Do not add other services. Langfuse is handled separately in Step 0.3.
Finish with verification commands.
```

**Verify:** `make up && make ping` → `PONG`

---

## Step 0.3 — Self-hosted Langfuse

```
Read CLAUDE.md and docs/PLAN.md (Step 0.3).

Implement Step 0.3 only: run Langfuse self-hosted with Docker on this Mac and prove tracing works.

Tasks:
1. infra/langfuse/README.md documenting the approach: use the official Langfuse docker compose
   (check the current official self-hosting docs for the correct repo and compose file;
   do not invent configuration). Clone it into infra/langfuse/vendor/ (gitignored).
2. Makefile targets: langfuse-up (clone if missing, then docker compose up -d from vendor dir),
   langfuse-down, langfuse-logs.
3. Check for host port conflicts with our services (app Redis is 6380, API 8080, LiteLLM 4000,
   llama.cpp 8081). Langfuse UI must be on 3000. If the official compose uses ports that conflict,
   document an override file instead of editing vendor files.
4. app/tracing/langfuse_client.py: a thin helper that creates a Langfuse client from settings and
   exposes a context manager for a trace and nested spans. Keep it minimal.
5. scripts/smoke_trace.py: sends one trace with two child spans and some metadata, then flushes.
6. Add the Langfuse keys placeholders to .env.example with instructions: open localhost:3000,
   create an organisation and project, create API keys, paste them into .env.

Finish with the exact manual steps I must do in the Langfuse UI and the verification commands.
```

**Verify:** `make langfuse-up`, create project + keys in UI, fill `.env`, run `uv run python scripts/smoke_trace.py`, see the trace in the UI.

---

## Step 0.4 — Local model + LiteLLM gateway

```
Read CLAUDE.md, docs/PRD.md (sections 9.3 and 9.5) and docs/PLAN.md (Step 0.4).

Implement Step 0.4 only.

Context: on the Mac we run a small Qwen instruct model with llama.cpp (installed via Homebrew).
The application must only talk to LiteLLM using the aliases po-fast, po-accurate, po-baseline.

Tasks:
1. infra/llamacpp/start.sh: starts llama-server on port 8081 with: --jinja, --parallel 4,
   --ctx-size 32768, model taken from env var LOCAL_GGUF (Hugging Face repo:quant format, used with -hf).
   Print a clear error if LOCAL_GGUF is not set. Add LOCAL_GGUF to .env.example with a comment
   recommending a ~7–9B Qwen instruct GGUF at Q4_K_M from the official Qwen organisation.
2. infra/litellm/config.dev.yaml: three model aliases (po-fast, po-accurate, po-baseline) all pointing to
   the local llama.cpp OpenAI-compatible endpoint. Set request timeout, num_retries 2.
   Also create infra/litellm/config.h100.yaml as a TEMPLATE (vLLM at http://localhost:8000/v1 via SSH tunnel,
   po-fast → MoE model, po-accurate → Qwen/Qwen3.8-27B, po-baseline → same 27B) with TODO comments;
   it will be finalised in Phase 7.
3. Makefile targets: model (runs start.sh), litellm (runs the proxy with config.dev.yaml on port 4000
   via uv run litellm), litellm-h100 (same with the h100 config).
4. scripts/smoke_llm.py: calls LiteLLM (OpenAI-compatible, using httpx or the openai client via LiteLLM base URL)
   with model po-fast, temperature 0, response_format json_schema for a tiny schema
   {"po_number": str, "po_date": str}, extra_body chat_template_kwargs enable_thinking false,
   on a short sample text. Print parsed JSON, latency in ms, prompt and completion tokens.
5. Document in README.md: how to start the model and LiteLLM, and how to switch to H100 later
   (only the LiteLLM config changes).

Do not write extraction logic. If a llama.cpp or LiteLLM flag you want to use is uncertain,
check its --help output or official docs rather than guessing.
Finish with verification commands.
```

**Verify:** terminal 1 `make model`, terminal 2 `make litellm`, terminal 3 `uv run python scripts/smoke_llm.py` → valid JSON, tokens and latency printed.

---

## Step 0.5 — Health check

```
Read CLAUDE.md and docs/PLAN.md (Step 0.5).

Implement Step 0.5 only: a `make doctor` command.

Tasks:
1. scripts/doctor.py checks, each with PASS/FAIL and a one-line fix hint:
   - .env present and required keys set
   - app Redis reachable on 6380
   - Langfuse UI reachable on 3000 and keys valid (auth check)
   - llama.cpp server reachable on 8081 (/health)
   - LiteLLM reachable on 4000 and each alias responds to a 1-token request
   - Docker running
   Print a summary table and exit non-zero if any check fails. Use only async httpx and redis client.
2. Makefile target: doctor.
3. Unit tests for the summary/exit-code logic (mock the checks).

Finish with verification commands.
```

**Verify:** `make doctor` → all PASS. Phase 0 complete — commit and tag `phase-0`.
