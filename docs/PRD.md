# PRD — Low-Latency Purchase Order Extraction System

**Project name:** `po-extractor`
**Owner:** Ketan Parmar
**Version:** 1.1 (incorporates "PO Data Extraction — Optimised Architecture", 6 Oct 2026)
**Timeline:** 3 days, 6+ hrs/day

---

## 1. Context

A self-hosted vision-language model (Qwen3.8-27B, BF16) served via vLLM behind LiteLLM on a single H100
currently takes **2–3 minutes per purchase order** (all pages sent as 120 dpi images in one request).
Root cause (most likely): Qwen3.8 runs in thinking mode by default at its deepest reasoning level, adding
thousands of hidden output tokens to a task that is reading, not reasoning. Secondary causes: BF16 weights,
oversized default context, untuned vLLM settings, verbose JSON output.

This project designs, builds and benchmarks an optimised system and proves every improvement with measurements.

## 2. Goals and non-goals

### Goals
1. Latency: **p95 < 30 s at 10 concurrent POs**; bursts of 20 stay within the 60 s acceptable ceiling.
2. **Accuracy first**: no optimisation is adopted unless evaluation proves accuracy is maintained.
3. Handle native and scanned PDFs equally well (≈ 50/50 mix, mixed pages possible).
4. **Only validated POs flow onward automatically**; everything else is retried once, then sent to review.
5. Observability (per-PO and per-call tracing, latency metrics) and reliability (retries, DLQ, idempotency).
6. **Confidentiality**: data never leaves the intranet; no telemetry from any component.
7. Reproducible benchmark on the target hardware (H100).

### Non-goals (v1)
- Fine-tuning; supplier-specific prompt hints (documented as phase 2).
- Real ERP integration (master-data checks run against a mock ERP master file).
- Production auth / multi-tenancy; Kubernetes.
- Handwritten POs.

## 3. Success metrics (SLOs)

| Metric | Target |
|---|---|
| End-to-end latency at 10 concurrent POs | **p95 < 30 s** |
| End-to-end latency at burst of 20 POs | p95 < 60 s (acceptable ceiling); report against 30 s too |
| Single PO, no load | p50 < 15 s native, < 20 s scanned |
| PO that needs a retry | completes (or reaches review) < 60 s |
| Field accuracy (normalised) | ≥ 97% |
| Critical-field document accuracy | ≥ 98% of POs |
| Silent errors on critical fields | 0 — failing POs must reach review, never pass as valid |
| Line-item recall | ≥ 99.5% |
| Throughput | 2,000 POs/day sustained with headroom |

End-to-end = upload accepted → result available (includes queue, pre-processing, LLM, validation, retry).

## 4. Users and use cases

- **Integrating system (ERP/AP):** submits PDFs via API, receives JSON via polling or webhook.
- **Reviewer:** fixes only the highlighted failed fields of POs in the review queue.
- **Operator:** monitors latency, traces, failures and DLQ in the dashboard.

## 5. Input specification

- PDFs of 2–4 pages (largest 5); **~50% native, ~50% scanned**; a single PO can mix both (decided per page).
- Indian GST purchase orders, no fixed format; **~30 frequent issuers account for ~70% of POs**.
- 30+ line items typical (up to ~80); tables span pages; optional terms & conditions pages.
- Max 10 MB, max 6 pages (configurable).

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

### 6.2 Line item fields (15 per row)
`line_no`\*, `item_code`, `description`\*, `hsn_sac`\*, `quantity`\*, `uom`, `unit_rate`\*, `discount_pct`, `taxable_value`\*, `gst_rate`\*, `cgst_amount`, `sgst_amount`, `igst_amount`, `line_total`\*, `line_delivery_date`

\* = critical field

### 6.3 LLM output contract (minimise output tokens)
- **Extract what is printed; compute what is derivable.** `COMPUTED_FIELDS` are never requested from the model:
  `total_tax` and per-line `cgst_amount`, `sgst_amount`, `igst_amount` (computed in code from taxable value,
  GST rate and supply type). Printed values needed for arithmetic checks — `taxable_value`, `line_total`,
  tax totals, `grand_total`, `amount_in_words` — are still extracted, because a check needs a printed value
  to compare against.
- **Short keys** in the LLM schema (e.g. `po_no`, `v_gstin`), mapped back to full names in code.
- **Absent fields omitted** (no nulls, notes, or "not found" text).
- **Line items as compact arrays**; column order is given once in the prompt.
- Compact JSON (no whitespace) where the serving stack allows it.

### 6.4 Result envelope
```json
{
  "job_id": "...", "file_sha256": "...",
  "status": "completed | needs_review | failed",
  "data": { "...header fields...": "...", "line_items": [ ... ] },
  "validation": { "passed": true, "checks": [ {"rule": "...", "passed": true, "fields": [], "detail": "..."} ] },
  "routing": { "strategy": "two_call | per_page", "retried_calls": [], "retry_mode": "thinking_low" },
  "pages": [ {"page": 1, "kind": "native | scanned"} ],
  "timings_ms": { "queue": 0, "preprocess": 0, "llm": {"H": 0, "LI": 0}, "validate": 0, "retry": 0, "total": 0 },
  "tokens": { "H": {"in": 0, "out": 0}, "LI": {"in": 0, "out": 0} }
}
```

## 7. Functional requirements

| ID | Requirement |
|---|---|
| FR-01 | `POST /v1/po` accepts a PDF, returns `job_id` immediately (async). |
| FR-02 | `GET /v1/po/{job_id}` returns status and result; optional webhook on completion. |
| FR-03 | Idempotency: file SHA-256 is the job ID; duplicates return the existing job. |
| FR-04 | Per-page classification: native (> 50 meaningful chars of text) or scanned. |
| FR-05 | Native pages → text layer (pymupdf4llm) **plus** page image; scanned pages → page image only. Image DPI configurable (default 150; baseline 120; ablation 120/150/200). Pages with no extractable fields (T&C) are skipped when detected. |
| FR-06 | Regex hints (GSTIN, PAN, email, phone, dates) from text layers, used for grounding checks. |
| FR-07 | Call strategy (configurable): `two_call` (header + line items, both see all pages), `per_page` (header + one line-item call per item page), **`adaptive` (default)**: two_call, switching line items to per_page when the PO is long (estimated rows > threshold or > 2 item pages). |
| FR-08 | Structured output: every call constrained by a JSON schema; temperature 0; presence penalty 0; repetition penalty 1.0; max_tokens safety cap. |
| FR-09 | Merge line items (per-page mode), repair rows split across page breaks, deduplicate; compute `COMPUTED_FIELDS`. |
| FR-10 | Validation engine runs all rules in §9.4. |
| FR-11 | Retry: only the failed call(s) are re-run **once** on `po-accurate` (same model, thinking on at low effort). |
| FR-12 | Still failing → `needs_review` with failed fields highlighted; review queue in dashboard. |
| FR-13 | Transient errors: timeout per call (120 s), retries with backoff; exhausted → DLQ. |
| FR-14 | Per-PO priority: calls of earlier POs are scheduled first (vLLM priority scheduling). |
| FR-15 | Global concurrency limit on in-flight LLM calls, aligned with vLLM `max-num-seqs` (LiteLLM limits never lower). |
| FR-16 | Live dashboard: upload, per-PO progress, call timeline, latency (p50/p95), native vs scanned split, review queue, DLQ, trace links. |
| FR-17 | Tracing (Langfuse self-hosted): trace per PO, span per stage and call, with tokens and latency. |
| FR-18 | Evaluation CLI: field, line-item and document-level scores, split by native / scanned / layout / issuer. |
| FR-19 | Benchmark CLI: concurrency sweep and bursts; raw results saved; report generated. |
| FR-20 | Baseline pipeline (current setup: 120 dpi images, single call, default thinking). |
| FR-21 | Synthetic dataset generator with exact ground truth, paired native/scanned versions, mock ERP masters. |

## 8. Non-functional requirements

- **Latency budget (single PO):** pre-processing ~1 s (render + text), network ~0.3 s, prefill ~1–2 s,
  slowest call ≤ 20 s, validation < 0.1 s; one retry must keep the PO under 60 s.
- **Accuracy:** §3; every potentially lossy change validated against the reference (§11.3).
- **Portability:** the app talks only to LiteLLM; local llama.cpp ↔ H100 vLLM is a config change.
- **Reliability:** every job ends as `completed`, `needs_review`, or in the DLQ. Nothing lost or processed twice.
- **Confidentiality:** self-hosted model and tracing; `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `HF_HUB_OFFLINE=1`
  after weights are downloaded; LiteLLM telemetry and external callbacks off; in production, outbound internet
  blocked at the firewall on the GPU server. No PO data in git.
- **Reproducibility:** seeded data generation; versioned configs; raw benchmark results saved.

## 9. Architecture

### 9.1 Flow
```
Client → FastAPI (POST /v1/po → job_id = file SHA-256)
       → Redis queue (arq) → async worker pool
          1. Pre-processing: per-page native/scanned classification; skip field-less pages
             native  → text layer (pymupdf4llm) + 150 dpi image
             scanned → 150 dpi image
          2. Regex hints from text layers
          3. Call planner (adaptive): H + LI   or   H + LI-p1 … LI-pN
          4. LiteLLM → vLLM (Qwen3.8-27B-FP8, thinking off, MTP speculative decoding)
          5. Merge + compute derived fields + validation
             all pass → completed
             any fail → re-run failed call(s) once on po-accurate (thinking low) → re-validate
             still fail → needs_review (failed fields highlighted)
       → SQLite store → result / webhook / SSE → Dashboard
       Traces → Langfuse (self-hosted)     Transient failures exhausted → DLQ
```

### 9.2 Message structure (prefix-cache friendly)
Fixed order for every call of a PO: `[system prompt] → [page content: per page, image then text layer if native]
→ [field definitions + task]`. All calls of one PO share the page-content prefix, so vLLM prefix caching
reads the pages once.

### 9.3 Why `adaptive` is the default strategy
- Typical POs: two calls keep request count low (40 requests at a burst of 20), maximise prefix reuse,
  and the line-item call is short enough.
- Long POs (many rows): the line-item call becomes the slowest part; splitting it per page caps worst-case
  latency at the cost of a few more requests.
- The benchmark measures `two_call`, `per_page` and `adaptive` side by side to justify the choice.

### 9.4 Validation rules
1. **Schema:** types valid; required (critical) fields present.
2. **GSTIN** format + checksum; first 2 digits = state code; PAN = GSTIN chars 3–12; PAN format.
3. **Tax regime:** intra-state (vendor state = place of supply) → CGST + SGST, IGST 0; inter-state → IGST only.
4. **Line arithmetic:** `taxable_value ≈ qty × rate × (1 − discount%)`; `line_total ≈ taxable + line taxes` (±1.00).
5. **Totals:** Σ taxable = `subtotal`; computed tax sums = printed tax totals;
   `grand_total = subtotal + taxes + freight + other − discount + round_off`, |round_off| ≤ 1.
6. **Continuity:** `line_no` continuous, no gaps or duplicates (detects rows lost at page breaks).
7. **Amount in words** parses to `grand_total`.
8. **Formats:** valid dates, `delivery_date ≥ po_date`, numeric HSN/SAC (4–8 digits), known currency codes.
9. **Grounding (native pages):** identifiers (GSTIN, PAN, PO number, item codes, HSN) appear in the text layer.
10. **ERP master data (mock):** vendor/buyer GSTIN exists in masters; item codes exist; PO number not already
    processed for that issuer (duplicate detection).

### 9.5 Model aliases (LiteLLM)
| Alias | Dev (Mac) | Benchmark / Prod (H100) |
|---|---|---|
| `po-fast` | Qwen3.5-9B (llama.cpp), thinking off | `Qwen/Qwen3.8-27B-FP8`, thinking off |
| `po-accurate` | same local model, thinking on | same FP8 model, thinking on, `reasoning_effort: low` |
| `po-baseline` | same local model, thinking default | `Qwen/Qwen3.8-27B` BF16, thinking default |
| `po-moe` (optional) | — | MoE candidate (e.g. Qwen3.6-35B-A3B FP8), thinking off |

### 9.6 vLLM server configuration (benchmark / production)
```bash
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --max-num-seqs 16 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 6, "video": 0}' \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --generation-config vllm --scheduling-policy priority
```
- KV cache kept at full precision (memory is not tight; avoid accuracy risk).
- `max-num-seqs` tuned by benchmark (16 → 24/32) for the burst of 20; LiteLLM limits must not be lower.
- If MTP conflicts with structured output in the installed vLLM version, drop that line (fallback: n-gram or none).
- Verify flags against the installed vLLM version and the official Qwen3.8 recipe before the run.

## 10. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Language / packaging | Python 3.12, `uv` | fast, reproducible |
| API | FastAPI + Uvicorn | async, OpenAPI, SSE |
| Queue | Redis 7 + `arq` | async jobs, retries; DLQ on top |
| Storage | SQLite (`aiosqlite`, SQLAlchemy) | zero-ops demo; Postgres for prod |
| PDF | PyMuPDF, `pymupdf4llm` | text layer, page classification, rendering |
| Schema / validation | Pydantic v2 | single source of truth → LLM JSON schemas |
| LLM gateway | LiteLLM proxy | aliases, retries, logging (telemetry off) |
| Inference (bench/prod) | vLLM on H100 (Lightning AI) | FP8, MTP speculative decoding, prefix cache, priority |
| Inference (dev) | llama.cpp `llama-server` on Mac (Qwen3.5-9B Q4_K_M) | OpenAI-compatible, vision, JSON schema |
| Tracing | Langfuse self-hosted | stays on-prem |
| Dashboard | FastAPI-served HTML, vanilla JS, Chart.js, SSE | no build step |
| Data generation | Faker (`en_IN`), Jinja2, Playwright, num2words, Pillow | realistic POs + scans with exact truth |
| Benchmark / reports | httpx, pandas, matplotlib | load generation, charts |
| Quality | ruff, pytest | lint, tests |

## 11. Data and evaluation strategy

### 11.1 Datasets
- **Synthetic (primary):** 60 native POs (6 layouts), 2–4 pages (5 allowed with T&C), 20–80 line items
  (mostly 30–50). **30 frequent issuers** (each bound to one layout and its own PO-number/date formats)
  produce ~70% of POs; the rest are one-off issuers.
- **Scanned twins:** every PO also gets a scanned version with identical truth → direct native-vs-scanned
  comparison and a realistic 50/50 benchmark mix. Plus ~10 mixed POs (some pages scanned).
- **Mock ERP masters:** suppliers/buyers (name, GSTIN), item codes, already-processed PO numbers
  (including a few deliberate duplicates in the dataset).
- **Splits:** dev 10 POs (+ twins), test 50 POs (+ twins), stratified by layout.
- **Northwind (HF):** smoke test. **FATURA (Zenodo, CC BY 4.0):** 50-image sample for scanned robustness.

### 11.2 Metrics
Field accuracy (normalised exact match), line-item precision/recall/F1, critical-field document accuracy,
validation pass rate, retry rate, review-queue rate — each split by native / scanned / layout / issuer type.

### 11.3 Acceptance rule for lossy changes
Adopt only if critical-field document accuracy drops ≤ 0.5 pp and field accuracy ≤ 1 pp versus the reference
(27B BF16, thinking off, same input) on the test split.

## 12. Benchmark methodology

| Config | Model | Input | Strategy | Notes |
|---|---|---|---|---|
| B0 Baseline | 27B BF16 | 120 dpi images | single call | thinking default (reproduces today) |
| C1 | 27B BF16 | hybrid, 150 dpi | two_call | thinking off, short schema, computed fields |
| C2 | 27B **FP8** | hybrid, 150 dpi | two_call | C1 + official FP8 checkpoint |
| C3 | 27B FP8 | hybrid, 150 dpi | two_call | C2 + MTP speculative decoding |
| C4 | 27B FP8 | hybrid, 150 dpi | **adaptive** | C3 + adaptive split (candidate recommendation) |
| C4-pp | 27B FP8 | hybrid, 150 dpi | per_page | strategy comparison |
| C5 (optional) | MoE FP8 | hybrid, 150 dpi | adaptive | latency/accuracy trade-off reference |

**Ablations (accuracy-focused, scanned subset):** 120 / 150 / 200 dpi; native text-only vs text+image.
**Load patterns:** concurrency 1, 5, **10 (SLO)**; burst of 20 with and without per-PO priority; document mix 50/50.
**Recorded:** per-PO latency (p50/p95/max) client-side and server-side, TTFT, output tokens, MTP acceptance
(`vllm:spec_decode_num_accepted_tokens_total`), requests waiting, accuracy, retry and review rates,
throughput, cost per 1,000 POs.

## 13. Infrastructure and sizing

- **Dev:** MacBook M5 Pro 24 GB — app, Redis, Langfuse (Docker), Qwen3.5-9B via llama.cpp.
- **Benchmark:** Lightning AI 1×H100 80 GB running vLLM only; Mac connects via SSH tunnel.
- **Production recommendation:** 1×H100 is expected to meet the SLO at 2,000 POs/day (≈ 1.4 POs/min average)
  if C4 meets p95 < 30 s at 10 concurrent and < 60 s at burst 20 — confirmed by measurement. A second H100
  is recommended for redundancy and burst headroom, not for the base SLO. All components on the intranet.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MTP + structured output incompatible in vLLM version | Drop spec decoding line; C2 config remains valid |
| Scanned accuracy lower than native | DPI ablation; thinking-low retry; review queue |
| Long POs exceed 30 s | Adaptive per-page line-item split |
| H100 availability / preemption | On-demand machine; launch scripts prepared in advance |
| Live demo failure | Pre-meeting checklist; recorded video; saved report |
| Mac memory pressure | Docker 8 GB; 9B Q4 model; stop unused services |
| Claude Code Pro limits | Small prompts; commit per step |

## 15. Assumptions
- Requirements confirmed: ~50% scanned, ~2,000 POs/day, typically 10 concurrent, bursts of 20, 2–4 pages.
- Field list (§6) defined by us for a typical Indian GST PO; the interviewer's real test set is not available,
  so the synthetic set stands in for it.
- Model IDs: `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.8-27B-FP8` (confirm before benchmark).

## 16. Deliverables
1. GitHub repository (code, configs, tests, README). 2. Synthetic dataset generator + evaluation harness.
3. Benchmark results (raw + report). 4. Live dashboard demo. 5. Interactive HTML presentation.
6. Design document (this PRD + decisions). 7. Recorded demo video (fallback).
