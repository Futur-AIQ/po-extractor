# Build Plan — po-extractor (v1.1)

9 phases, 37 steps, 3 days. Each step = one Claude Code prompt (in `docs/prompts/phase-N.md`).
Every step ends with tests passing and a git commit.
v1.1 changes: hybrid native/scanned input, adaptive call strategy, computed fields + short keys,
thinking-low retry, mock ERP masters, issuer pool, MTP speculative decoding, confidentiality settings.

## Schedule

| Day | Phases | Outcome by end of day |
|---|---|---|
| Day 1 | 0 Setup · 1 Schema & synthetic data · 2 Baseline & evaluation | 60 POs + scanned twins with truth; baseline scored locally |
| Day 2 | 3 Extraction core · 4 Service layer · 5 Dashboard | Optimised pipeline end to end on Mac with live dashboard |
| Day 3 | 6 Benchmark tooling · 7 H100 runs · 8 Presentation | Measured H100 results, presentation, repo submitted |

Cut order if behind: C5 MoE → FATURA sample → C4-pp → webhook → DPI 200 ablation.

---

## Phase 0 — Foundations ✅ (unchanged)
0.1 scaffold · 0.2 Redis · 0.3 Langfuse · 0.4 local model + LiteLLM · 0.5 doctor.
Note: `infra/litellm/config.h100.yaml` is a template; it is rewritten in Step 7.1 for the v1.1 aliases.

## Phase 1 — Schema & synthetic data

| Step | Objective | Acceptance criteria |
|---|---|---|
| 1.1 ✅ | Pydantic schema, groups, JSON schemas | done |
| 1.2 ✅ | Indian data providers | done |
| 1.3 ✅ | PO generator + consistency check | done |
| 1.4 ✅ | Six layouts | done |
| 1.4a | Schema patch: computed fields, short LLM keys, header schema (all groups), line-item schema without computed columns | Key map round-trips; computed fields absent from LLM schemas |
| 1.4b | Generator patch: 30-issuer pool (70%), 20–80 lines, mock ERP masters, deliberate duplicates | Issuer share ≈ 70%; masters cover all truth parties/items; consistency tests still pass |
| 1.5 | PDF rendering + dataset build (2–4 pages, 5 with T&C; layout per issuer) | `make dataset` → 60 PDFs + truth + masters + splits + preview |
| 1.6 | Scanned twins for all POs + 10 mixed POs + public downloads | Twins have no text layer; mixed have both; downloads verified |

## Phase 2 — Baseline & evaluation harness

| Step | Objective | Acceptance criteria |
|---|---|---|
| 2.1 | Normalisation + scoring (field, line-item P/R/F1, critical-doc accuracy; splits by native/scanned/layout/issuer) | Truth vs itself = 100% |
| 2.2 | Baseline B0: all pages as 120 dpi images, single call, default thinking | Runs on dev split; outputs + timings + tokens saved |
| 2.3 | Evaluation CLI + report | `make eval RUN=<dir>` prints and saves report |

## Phase 3 — Optimised extraction core

| Step | Objective | Acceptance criteria |
|---|---|---|
| 3.1 | Pre-processing: per-page native/scanned classification, text layer + image (DPI configurable), field-less page skipping | Twins classified correctly; T&C pages skipped |
| 3.2 | Regex hints + grounding index (native pages) | Finds all GSTINs/PANs in native truth |
| 3.3 | Message builder: prefix-stable order (pages first, task last), header and line-item tasks, short keys, compact rows | All calls of a PO share identical prefix (test) |
| 3.4 | LLM client via LiteLLM: schema-constrained, temperature 0, penalties neutral, max_tokens cap, timeout, tokens + timings | Parsed outputs mapped to full field names |
| 3.5 | Call planner (`two_call` / `per_page` / `adaptive`) + orchestrator (gather, global semaphore, PO priority) | Strategy selectable; calls concurrent (mocked timing test) |
| 3.6 | Merge + page-break repair + computed fields + validation engine (PRD §9.4, incl. mock ERP masters) | Unit test per rule; truth passes 100% |
| 3.7 | Retry failed call(s) once on `po-accurate` (thinking low) → re-validate → `needs_review` with failed fields | Injected failure → only that call retried |
| 3.8 | Optimised pipeline CLI on dev split (native + scanned) + evaluation vs baseline | Report shows both; latency + tokens logged |

## Phase 4 — Service layer
| Step | Objective | Acceptance criteria |
|---|---|---|
| 4.1 | SQLite store (jobs, results, events, timings, review queue) | CRUD tests |
| 4.2 | FastAPI: submit, status/result, SHA-256 idempotency, webhook | Duplicate upload → same job |
| 4.3 | arq worker: timeouts, backoff, DLQ, PO priority, global concurrency | Forced failure → DLQ |
| 4.4 | Langfuse tracing (trace per PO, spans per stage/call, tokens) — no external callbacks | Trace shows all calls |
| 4.5 | SSE progress events | Live stage events in browser |

## Phase 5 — Live dashboard
| Step | Objective | Acceptance criteria |
|---|---|---|
| 5.1 | Upload + live PO list (stage progress, native/scanned badges) | 5 uploads progress live |
| 5.2 | PO detail: call timeline (Gantt), extracted data, validation checks, failed fields highlighted | Click PO → full view |
| 5.3 | Latency charts (p50/p95), review queue, DLQ, trace links | Updates live during a burst |

## Phase 6 — Benchmark tooling (on Mac)
| Step | Objective | Acceptance criteria |
|---|---|---|
| 6.1 | Load generator: sweep (1/5/10), burst 20 (priority on/off), 50/50 native-scanned mix, JSONL results | Runs locally |
| 6.2 | vLLM metrics collector (TTFT, latency, waiting, MTP acceptance) + client/server split | Works on mock endpoint |
| 6.3 | Matrix runner (B0, C1–C5, ablations) + report (charts, tables, cost per 1k POs) | Report from local run |

## Phase 7 — H100 runs (~4–5 GPU hours)
| Step | Objective | Acceptance criteria |
|---|---|---|
| 7.1 | vLLM launch scripts (BF16, FP8, FP8+MTP, MoE), confidentiality env vars, Lightning setup, SSH tunnel, LiteLLM H100 config with v1.1 aliases | Reviewed; dry-run on Mac |
| 7.2 | Run matrix + ablations; save raw results; shut down GPU | All configs measured |
| 7.3 | Final evaluation, report, recommendation | Measured numbers + decision |

## Phase 8 — Presentation & submission
| Step | Objective | Acceptance criteria |
|---|---|---|
| 8.1 | README + design decisions doc | Reproducible setup |
| 8.2 | Interactive HTML presentation (results first) | Opens offline |
| 8.3 | Demo script, recorded video, pre-meeting checklist | GPU warm-up + fallback covered |

---

## Claude Code working rules (Pro plan)
1. `/clear` before each step; paste the step prompt.
2. **[plan]** steps: plan mode first (Shift+Tab), review, then execute.
3. One step per prompt; commit after each.
4. Run acceptance commands yourself before committing.
5. If Claude Code drifts: "Re-read docs/PRD.md section X and align."
6. Secrets only in `.env`.
