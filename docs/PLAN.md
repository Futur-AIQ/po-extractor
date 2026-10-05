# Build Plan — po-extractor

9 phases, 34 steps, 3 days. Each step = one Claude Code prompt (in `docs/prompts/phase-N.md`).
Every step ends with tests passing and a git commit.

## Schedule

| Day | Phases | Outcome by end of day |
|---|---|---|
| Day 1 | 0 Setup · 1 Schema & synthetic data · 2 Baseline & evaluation | Dataset of 60 POs with ground truth; baseline scored locally |
| Day 2 | 3 Extraction core · 4 Service layer · 5 Dashboard | Full optimised pipeline running end to end on Mac with live dashboard |
| Day 3 | 6 Benchmark tooling · 7 H100 runs · 8 Presentation & submission | Measured H100 results, presentation, repo submitted |

Buffer: Day 3 evening. If behind schedule, cut in this order: C5 two-tier routing → FATURA sample → webhook → stress burst of 50.

---

## Phase 0 — Foundations (Day 1, ~2 h)

| Step | Objective | Acceptance criteria |
|---|---|---|
| 0.1 | Repo scaffold: uv project, folder structure, settings, lint, tests | `uv run pytest` and `uv run ruff check .` pass; settings load from `.env` |
| 0.2 | Docker Compose for app Redis (port 6380) + Makefile | `make up` starts Redis; `make ping` returns PONG |
| 0.3 | Self-hosted Langfuse + smoke trace | Langfuse UI at localhost:3000; a test trace appears |
| 0.4 | Local model (llama.cpp) + LiteLLM proxy with aliases | Structured JSON call via `po-fast` alias succeeds with thinking off |
| 0.5 | `make doctor` health check | One command reports status of Redis, Langfuse, LiteLLM, model |

## Phase 1 — Schema & synthetic data (Day 1, ~3 h)

| Step | Objective | Acceptance criteria |
|---|---|---|
| 1.1 | Pydantic schema for PO (58 header + 15 line fields), group definitions, JSON schema export | Schema → JSON schema per group; unit tests |
| 1.2 | Indian data providers: states/codes, valid GSTIN with checksum, PAN, HSN list, UoM, GST slabs, amount in words (lakh/crore) | GSTIN checksum tests; arithmetic consistent |
| 1.3 | PO data generator: realistic, internally consistent POs + difficulty knobs, seeded | 100 generated POs pass all §9.4 validation rules |
| 1.4 | 6 HTML layouts (Jinja2) with page-spanning tables | Each layout renders visibly different structure |
| 1.5 | PDF rendering (Playwright) + ground truth JSON + dataset build CLI | `make dataset` creates 60 PDFs (2–3 pages) + truth files; splits dev/test |
| 1.6 | Scanned variants (rasterise + noise) + public dataset download scripts (Northwind, FATURA sample) | 10 image-only PDFs; scripts download and verify checksums |

## Phase 2 — Baseline & evaluation harness (Day 1, ~2 h)

| Step | Objective | Acceptance criteria |
|---|---|---|
| 2.1 | Normalisation + scoring (field, line-item P/R/F1, critical-field doc accuracy) | Scoring ground truth against itself = 100% |
| 2.2 | Baseline pipeline B0 (page images, single request, thinking configurable) | Runs on dev split, saves outputs + timings |
| 2.3 | Evaluation CLI + markdown/JSON report | `make eval RUN=<dir>` prints and saves report |

## Phase 3 — Optimised extraction core (Day 2, ~3.5 h)

| Step | Objective | Acceptance criteria |
|---|---|---|
| 3.1 | Text extraction (pymupdf4llm), page markers, quality check, image fallback trigger | Digital pages → markdown; scanned pages flagged |
| 3.2 | Regex hints + grounding index | Finds all GSTINs/PANs/emails in synthetic set |
| 3.3 | Prompt builder (prefix-stable) + group tasks + compact line-item format | Same prefix bytes across all groups of a PO (test) |
| 3.4 | LLM client via LiteLLM: schema-constrained calls, thinking off, timings, token usage | Returns parsed Pydantic objects; retries on transient errors |
| 3.5 | Fan-out orchestrator: asyncio.gather, global semaphore, priority field | Groups run concurrently (timing test with mocked client) |
| 3.6 | Merge + page-break row repair + validation engine (all §9.4 rules) | Unit tests per rule; synthetic truth passes 100% |
| 3.7 | Targeted re-ask + two-tier routing + review flag | Injected failure → only failed group re-asked on `po-accurate` |
| 3.8 | Optimised pipeline CLI on dev split + evaluation | Accuracy ≥ baseline on local model; latency logged |

## Phase 4 — Service layer (Day 2, ~2 h)

| Step | Objective | Acceptance criteria |
|---|---|---|
| 4.1 | SQLite store (jobs, results, events, timings) | CRUD tests pass |
| 4.2 | FastAPI endpoints: submit, status/result, idempotency, webhook | Swagger works; duplicate upload returns same job |
| 4.3 | arq worker: retries with backoff, DLQ, PO-priority, global concurrency | Forced failure lands in DLQ after N attempts |
| 4.4 | Langfuse tracing (trace per PO, span per stage/sub-request) | Trace shows all sub-requests with tokens and latency |
| 4.5 | SSE event stream for live progress | Browser receives stage events in real time |

## Phase 5 — Live dashboard (Day 2, ~1.5 h)

| Step | Objective | Acceptance criteria |
|---|---|---|
| 5.1 | Upload panel + live PO list with stage progress | Upload 5 POs → see each progress live |
| 5.2 | Per-PO timeline (Gantt of sub-requests) + result viewer with validation checks | Click PO → timeline + extracted JSON + checks |
| 5.3 | Latency charts (p50/p95, histogram), review queue, DLQ panel, Langfuse links | Charts update live during a burst |

## Phase 6 — Benchmark tooling (Day 3 morning, ~2 h, on Mac)

| Step | Objective | Acceptance criteria |
|---|---|---|
| 6.1 | Load generator: concurrency sweep + burst mode (priority on/off), raw results to JSONL | Runs against local stack |
| 6.2 | vLLM metrics collector (TTFT, request latency, queue, KV usage) + client/server latency split | Works against a mock Prometheus endpoint |
| 6.3 | Benchmark matrix runner (configs B0, C1–C5) + report generator (charts, tables, cost per 1k POs) | Report HTML generated from local run |

## Phase 7 — H100 runs (Day 3, ~4–5 GPU hours)

| Step | Objective | Acceptance criteria |
|---|---|---|
| 7.1 | vLLM launch scripts per config + Lightning setup script + SSH tunnel + LiteLLM H100 config (prepared on Mac before starting GPU) | Scripts reviewed; dry-run on Mac |
| 7.2 | Run matrix on H100, save raw results, shut down GPU | All configs measured; results committed |
| 7.3 | Final evaluation + report + decision (which config is recommended) | Report with measured numbers and recommendation |

## Phase 8 — Presentation & submission (Day 3 evening, ~2 h)

| Step | Objective | Acceptance criteria |
|---|---|---|
| 8.1 | README (setup, run, architecture, results) + design decisions doc | New machine can follow README |
| 8.2 | Interactive HTML presentation (results-first storyline, charts embedded) | Opens offline in a browser |
| 8.3 | Demo script, recorded demo video, pre-meeting checklist | Checklist covers GPU warm-up and fallback |

---

## Claude Code working rules (Pro plan)

1. Start each step with `/clear`, then paste the step prompt. The prompt tells Claude Code which docs to read.
2. For steps marked **[plan]**, use plan mode first (Shift+Tab), review the plan, then let it execute.
3. One step per prompt. If the usage limit is close, finish and commit the current step rather than starting the next.
4. After each step: run the acceptance command yourself, then `git commit`.
5. If Claude Code drifts from the PRD, say: "Re-read docs/PRD.md section X and align."
6. Never paste secrets into prompts; keep them in `.env` (gitignored).

## Manual prerequisites (before Step 0.1)

- Docker Desktop running, memory limit set to **8 GB** (Langfuse self-host needs more than Redis alone).
- Homebrew, git, GitHub repo `po-extractor` created (private until submission).
- `uv` installed: `brew install uv`.
- `llama.cpp` installed: `brew install llama.cpp`.
- Claude Code installed and logged in; VS Code open on the empty repo.
- Copy `CLAUDE.md` to repo root and `docs/` folder into the repo, then commit.
