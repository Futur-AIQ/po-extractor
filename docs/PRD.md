# PRD — Low-Latency Purchase Order Extraction System

**Project name:** `po-extractor`
**Owner:** Ketan Parmar
**Status:** Approved for build (v1.0)
**Timeline:** 3 days, 6+ hrs/day

---

## 1. Context

A self-hosted vision-language model (Qwen3.8-27B, full precision) served via vLLM on a single H100 currently takes **2–3 minutes per purchase order** (2–3 page PDF sent as page images). The business needs to process many POs in parallel with high accuracy and low latency.

This project designs, builds, and benchmarks an optimised extraction system and proves the improvement with measured results.

## 2. Goals and non-goals

### Goals
1. Reduce end-to-end latency per PO from ~150 s to **p95 < 30 s**, including queue time, at design load.
2. Keep **accuracy first**: no optimisation is adopted unless evaluation proves accuracy is maintained.
3. Handle **parallel bursts** of POs predictably (design burst 20, stress test 50).
4. Make the system **observable** (per-PO and per-request tracing, latency metrics) and **reliable** (retries, dead-letter queue, human review queue).
5. Prove all claims with a **reproducible benchmark** on the target hardware (H100).

### Non-goals (v1)
- Fine-tuning a model (documented as a roadmap item).
- Production authentication / multi-tenancy (API key only).
- Handwritten POs.
- Kubernetes / autoscaling implementation (documented as a recommendation).

## 3. Success metrics (SLOs)

| Metric | Target |
|---|---|
| End-to-end latency, single PO | p50 < 15 s |
| End-to-end latency at burst of 20 POs | p95 < 30 s (measured on 1×H100, projected for 2×H100) |
| Field accuracy (all fields, normalised) | ≥ 97% |
| Critical-field document accuracy (every critical field correct) | ≥ 98% of POs |
| Silent errors on critical fields | 0 — failing POs must be flagged for review, never passed as valid |
| Line-item recall (no missed rows) | ≥ 99.5% |
| Throughput | ≥ 2,000 POs/day sustained with headroom |

"End-to-end" = from upload accepted to result available (includes queue, extraction, LLM, validation, re-ask).

## 4. Users and use cases

- **Integrating system** (ERP / AP automation): submits PDFs via API, receives JSON via polling or webhook.
- **Operations reviewer**: works the human review queue for POs that fail validation.
- **Engineer / operator**: monitors latency, traces, failures, and DLQ in the dashboard.

## 5. Input specification

- Digital (text-selectable) PDFs, typically 2–3 pages; scanned PDFs are a supported minority via fallback.
- Indian GST purchase orders, many varied layouts (different vendors' ERP formats).
- 30+ line items per PO, tables that span pages; optional terms & conditions page.
- Max file size 10 MB, max 6 pages (configurable).

## 6. Output specification

### 6.1 Header fields (58)

**Document (10):** `po_number`\*, `po_date`\*, `amendment_no`, `amendment_date`, `quotation_ref`, `quotation_date`, `indent_no`, `currency`, `delivery_date`, `po_validity_date`

**Buyer (9):** `buyer_name`\*, `buyer_address`, `buyer_gstin`\*, `buyer_pan`, `buyer_state`, `buyer_state_code`, `buyer_contact_person`, `buyer_phone`, `buyer_email`

**Vendor (10):** `vendor_name`\*, `vendor_code`, `vendor_address`, `vendor_gstin`\*, `vendor_pan`, `vendor_state`, `vendor_state_code`, `vendor_contact_person`, `vendor_phone`, `vendor_email`

**Bill-to (4):** `bill_to_name`, `bill_to_address`, `bill_to_gstin`, `bill_to_state_code`

**Ship-to (4):** `ship_to_name`, `ship_to_address`, `ship_to_gstin`, `ship_to_state_code`

**Terms (8):** `payment_terms`, `delivery_terms`, `freight_terms`, `mode_of_transport`, `place_of_supply`, `delivery_location`, `warranty_terms`, `packing_instructions`

**Totals (11):** `subtotal`\*, `discount_total`, `freight_charges`, `other_charges`, `cgst_total`\*, `sgst_total`\*, `igst_total`\*, `total_tax`, `round_off`, `grand_total`\*, `amount_in_words`

**Approval (2):** `prepared_by`, `approved_by`

\* = critical field

### 6.2 Line item fields (15 per row)

`line_no`\*, `item_code`, `description`\*, `hsn_sac`\*, `quantity`\*, `uom`, `unit_rate`\*, `discount_pct`, `taxable_value`\*, `gst_rate`\*, `cgst_amount`, `sgst_amount`, `igst_amount`, `line_total`\*, `line_delivery_date`

### 6.3 Result envelope

```json
{
  "job_id": "...", "file_sha256": "...",
  "status": "completed | needs_review | failed",
  "data": { "...header fields...": "...", "line_items": [ ... ] },
  "validation": { "passed": true, "checks": [ {"rule": "...", "passed": true, "detail": "..."} ] },
  "routing": { "model_path": "fast | fast+fallback | accurate", "reasks": 0 },
  "timings_ms": { "queue": 0, "extract_text": 0, "llm": {"G1": 0, "G2": 0, "LI-p1": 0}, "validate": 0, "total": 0 }
}
```

## 7. Functional requirements

| ID | Requirement |
|---|---|
| FR-01 | `POST /v1/po` accepts a PDF, returns `job_id` immediately (async). |
| FR-02 | `GET /v1/po/{job_id}` returns status and result; optional webhook callback on completion. |
| FR-03 | Idempotency: identical file (SHA-256) returns the existing job/result. |
| FR-04 | Text extraction per page with a quality check; pages failing the check use the image fallback path. |
| FR-05 | Regex pre-extraction of pattern fields (GSTIN, PAN, email, phone, dates) used as hints and as grounding checks. |
| FR-06 | Fan-out: each PO is split into parallel sub-requests (header+terms, parties, totals, line items per page). |
| FR-07 | Structured output: every LLM call is constrained by a JSON schema. |
| FR-08 | Merge per-page line items, repair rows split across page breaks, deduplicate. |
| FR-09 | Validation engine runs business rules (§9.4); failures trigger targeted re-ask of the failed group only. |
| FR-10 | Two-tier routing: fast model first; failed groups re-asked on the accurate model. |
| FR-11 | POs still failing after fallback are marked `needs_review` and placed in the review queue. |
| FR-12 | Retries with exponential backoff for transient errors; after max attempts the job goes to the DLQ. |
| FR-13 | Per-PO priority: sub-requests of earlier POs are scheduled first (FIFO by PO arrival). |
| FR-14 | Global concurrency limit on in-flight LLM requests (configurable). |
| FR-15 | Live dashboard: upload, per-PO progress, sub-request timeline, latency charts (p50/p95), review queue, DLQ, link to trace. |
| FR-16 | Tracing: every PO is a trace; every sub-request a span with model, tokens, latency. |
| FR-17 | Evaluation CLI: scores outputs against ground truth (field, line-item, document level). |
| FR-18 | Benchmark CLI: concurrency sweep and burst tests; saves raw results and generates a report. |
| FR-19 | Baseline pipeline (naive: images, one request, thinking on) for before/after comparison. |
| FR-20 | Synthetic dataset generator with exact ground truth. |

## 8. Non-functional requirements

- **Latency:** see §3. Latency budget per PO: extraction ~1 s, network ~0.3 s, prefill ~1 s, slowest sub-request ≤ 20 s, validation < 0.1 s, one targeted re-ask must still fit in 30 s.
- **Accuracy:** see §3. Every potentially lossy optimisation must be validated against the baseline (§11).
- **Portability:** the application talks only to LiteLLM; switching local llama.cpp ↔ vLLM on H100 is a config change.
- **Reliability:** no lost jobs; every job ends in `completed`, `needs_review`, or DLQ.
- **Observability:** Langfuse traces (self-hosted), structured JSON logs, timing records per stage.
- **Security & privacy:** POs contain commercial data → fully self-hosted model, no third-party LLM APIs; vLLM endpoint protected by API key and SSH tunnel; no PO data in git.
- **Reproducibility:** fixed random seeds for data generation; benchmark configs version-controlled; raw results saved.

## 9. Architecture

### 9.1 Flow

```
Client → FastAPI (POST /v1/po → job_id)
       → Redis queue (arq) → Worker
          1. Text extraction (pymupdf4llm) + page quality check
             └─ failing pages → image fallback (120 DPI render)
          2. Regex hints (GSTIN, PAN, email, phone, dates)
          3. Fan-out orchestrator (asyncio.gather, global semaphore, PO priority)
             ├─ G1 header + terms
             ├─ G2 parties (buyer, vendor, bill-to, ship-to)
             ├─ G3 totals
             └─ LI-p1 … LI-pN  line items per page
          4. LiteLLM gateway → vLLM (fast model) / vLLM (accurate model)
          5. Merge + validation → targeted re-ask (accurate model) → review flag
       → SQLite store → result / webhook / SSE events → Dashboard
       Traces → Langfuse (self-hosted)    Failures → DLQ
```

### 9.2 Prompt structure (prefix-cache friendly)

Every sub-request uses the same order so vLLM prefix caching reuses the document:
`[system prompt] → [schema rules] → [full PO text, pages marked] → [group-specific task]`

### 9.3 Output token minimisation

- Thinking mode disabled (`chat_template_kwargs.enable_thinking=false`).
- JSON schema enforced; null/empty fields omitted.
- Line items as compact arrays: `{"cols": [...], "rows": [[...], ...]}`.
- Temperature 0 (greedy).

### 9.4 Validation rules

1. GSTIN format + checksum; first 2 digits = state code; PAN = GSTIN chars 3–12.
2. Intra-state supply (vendor state = place of supply) → CGST + SGST, IGST = 0; inter-state → IGST only.
3. CGST = SGST per line, each = half of GST rate.
4. `taxable_value ≈ quantity × unit_rate × (1 − discount_pct/100)` (tolerance ±1.00).
5. Σ line taxable values = `subtotal`; Σ line taxes = tax totals.
6. `grand_total = subtotal + taxes + freight + other − discount + round_off`; |round_off| ≤ 1.
7. `line_no` continuous, no gaps or duplicates (detects missed rows at page breaks).
8. `amount_in_words` parses to `grand_total`.
9. Dates valid; `delivery_date ≥ po_date`.
10. **Grounding check:** every extracted identifier (GSTIN, PAN, PO number, item codes, HSN) must appear in the source text (normalised). Prevents hallucinated values.

### 9.5 Model routing

| Alias | Dev (Mac) | Benchmark / Prod (H100) |
|---|---|---|
| `po-fast` | small local Qwen (llama.cpp) | MoE candidate (e.g. Qwen3.6-35B-A3B, FP8) |
| `po-accurate` | same local model | Qwen3.8-27B (FP8 or BF16) |
| `po-baseline` | same local model | Qwen3.8-27B BF16, images, thinking on |

If the MoE fails the accuracy bar, `po-fast` is pointed at the 27B (single-tier).

## 10. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Language / packaging | Python 3.12, `uv` | fast, reproducible environments |
| API | FastAPI + Uvicorn | async, OpenAPI docs, SSE support |
| Queue | Redis 7 + `arq` | async-native jobs, retries; DLQ implemented on top |
| Storage | SQLite (`aiosqlite`, SQLAlchemy) | zero-ops for demo; Postgres recommended for prod |
| PDF | PyMuPDF, `pymupdf4llm` | fast text + markdown tables; page rendering for fallback |
| Schema / validation | Pydantic v2 | single source of truth → JSON schema for LLM |
| LLM gateway | LiteLLM proxy | aliases, retries, fallbacks, logging |
| Inference (bench/prod) | vLLM on H100 (Lightning AI) | continuous batching, FP8, prefix cache, speculative decoding, priority scheduling |
| Inference (dev) | llama.cpp `llama-server` on Mac | OpenAI-compatible, parallel slots, JSON schema |
| Tracing | Langfuse (self-hosted, Docker) | traces/spans per PO and sub-request |
| Dashboard | Single HTML page served by FastAPI, vanilla JS, Chart.js, SSE | no frontend build step |
| Data generation | Faker (`en_IN`), Jinja2, Playwright (Chromium PDF), num2words | realistic Indian POs with exact ground truth |
| Benchmark / reports | httpx (async), pandas, matplotlib | load generation and charts |
| Quality | ruff, pytest | lint and tests |

## 11. Data and evaluation strategy

### 11.1 Datasets
- **Synthetic (primary):** ~60 POs across 6 layouts, 30–60 line items, 2–3 pages, Indian GST. Difficulty knobs: rows split across pages, multi-line descriptions, T&C page, missing optional fields, discounts/freight/round-off, intra- vs inter-state. ~10 "scanned" variants (rasterised + noise) for the fallback path. Split: 10 dev / 50 test.
- **Northwind (HF):** smoke test.
- **FATURA (Zenodo, CC BY 4.0):** 50-image sample for the image fallback path.

GST rate slabs are configurable; default to the current structure (verify before generating).

### 11.2 Metrics
Field accuracy (normalised exact match), line-item precision/recall/F1 (row matching on `line_no` then content), critical-field document accuracy, validation pass rate, review-queue rate. Normalisation: whitespace/case, numbers to Decimal, dates to ISO, GSTIN uppercase.

### 11.3 Acceptance rule for lossy optimisations
An optimisation is adopted only if critical-field document accuracy drops by ≤ 0.5 percentage points and field accuracy by ≤ 1 point versus the 27B BF16 reference on the test split.

## 12. Benchmark methodology

**Matrix (H100):**

| Config | Model | Input | Precision | Extras |
|---|---|---|---|---|
| B0 Baseline | 27B | images | BF16 | thinking on, single request |
| C1 | 27B | text | BF16 | fan-out, schema, thinking off |
| C2 | 27B | text | FP8 | C1 + FP8 |
| C3 | 27B | text | FP8 | C2 + n-gram speculative decoding |
| C4 | MoE | text | FP8 | fan-out, schema, thinking off |
| C5 | MoE → 27B | text | FP8 | two-tier routing |

**Load patterns:** concurrency sweep (1, 5, 10, 20), burst of 20 (with and without per-PO priority), stress burst of 50.
**Recorded:** per-PO end-to-end latency (p50/p95/max), server-side TTFT and request latency (vLLM metrics), output tokens, throughput (POs/min), accuracy, cost per 1,000 POs.
**Network:** client on Mac, server in cloud; report client-side and server-side latency separately.

## 13. Infrastructure and sizing

- **Dev:** MacBook M5 Pro 24 GB — app, Redis, Langfuse (Docker), small local model.
- **Benchmark:** Lightning AI 1×H100 80 GB running vLLM only; Mac connects via SSH tunnel.
- **Production recommendation:** 2×H100 — either MoE (fast) + 27B (fallback), or 2× 27B replicas with speculative decoding behind LiteLLM load balancing. Scale by adding replicas when queue wait or KV-cache usage stays high.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MoE accuracy below bar | Single-tier 27B configuration (C3) remains the recommendation |
| vLLM feature incompatibility (spec decoding + structured output) | Test early on H100; fall back to C2 |
| H100 availability / preemption | On-demand (not interruptible) for benchmark; launch scripts ready in advance |
| Live demo failure | Pre-meeting checklist; recorded demo video; saved benchmark report |
| Mac memory pressure (Langfuse + model) | Docker memory 8 GB; small local model; stop unused services |
| Claude Code Pro limits | Small, self-contained prompts; commit after each step |
| Rows split at page breaks | Merge-repair step + `line_no` continuity validation |

## 15. Assumptions

- Volume ~2,000 POs/day, bursts of 20; the interviewer did not specify.
- Latency target interpreted as end-to-end p95 < 30 s at design load.
- Field list defined by us (§6) to represent a typical Indian GST PO.
- Model ID for the 27B: `Qwen/Qwen3.8-27B` (confirm before benchmark).

## 16. Deliverables

1. GitHub repository (code, configs, tests, README).
2. Synthetic dataset generator and evaluation harness.
3. Benchmark results (raw + report with charts).
4. Live dashboard demo.
5. Interactive HTML presentation.
6. Design document (this PRD + architecture decisions).
7. Recorded demo video (fallback for the live meeting).
