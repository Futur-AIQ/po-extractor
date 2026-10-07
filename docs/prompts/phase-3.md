# Phase 3 — Optimised extraction core (Claude Code prompts) — v1.1

Goal: the optimised pipeline from PRD v1.1 — per-page native/scanned handling, prefix-stable messages,
compact output, adaptive call planning, validation, and a targeted thinking-low retry — runnable as a CLI
that writes the standard run format, so it can be scored and compared against the baseline.

All code lives in `app/extraction/`. Order: 3.1 → 3.8. `/clear` before each prompt; commit after each step.

**Before Step 3.8**, restart the local model with more parallel slots and context:
`PARALLEL=4 CTX_SIZE=65536 make model` (≈ 16k tokens per slot — enough for 4 page images + output).

---

## Step 3.1 — Pre-processing: page classification, text layer, images

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 5, 7 (FR-04, FR-05) and 9.1. Read the dataset truth meta
(page kinds, fieldless_pages) produced in Phase 1.

Implement Step 3.1 only: app/extraction/preprocess.py.

Tasks:
1. Data model: PageContent(page_no, kind: "native" | "scanned", text_md: str | None, image_b64: str,
   image_mime, width, height, skipped: bool, skip_reason: str | None) and PreparedDoc(doc_sha256,
   page_count, pages, timings_ms).
2. Per-page classification: native if the page's text layer has more than NATIVE_MIN_CHARS meaningful
   characters (letters/digits, default 50, from settings); otherwise scanned. Decided per page (mixed PDFs).
3. Native pages: text via pymupdf4llm (per-page markdown, tables preserved) AND a page image.
   Scanned pages: page image only. Image DPI from settings (default 150); image format configurable
   (JPEG quality 92 default, PNG option); cap the longest side at MAX_IMAGE_PX (default 2400) as a guard.
4. Field-less page skipping (setting, default on): only for NATIVE pages, only when confident — e.g. the
   page contains a terms-and-conditions heading and numbered clauses, and contains none of: GSTIN pattern,
   currency amounts table, line-item table. Never skip page 1. Never skip scanned pages.
5. CPU work runs off the event loop (asyncio.to_thread); prepare(pdf_bytes, settings) is async.
6. Native mode option for ablations: native_mode = "text+image" (default) | "text" | "image".
7. Tests against data/synthetic: classification matches truth meta for N, S and M variants (100%);
   T&C pages skipped where truth marks them field-less and no other page is skipped; images have the
   expected pixel size for the DPI; timing recorded.

Finish with a summary and verification commands.
```

**Verify:** `make test`

---

## Step 3.2 — Regex hints and grounding index

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 7 (FR-06) and 9.4 (rules 2, 8, 9). Read
app/extraction/preprocess.py and datagen/india.py (GSTIN checksum helpers — reuse, do not duplicate;
move shared helpers to a common module such as app/common/gst.py if needed, updating imports).

Implement Step 3.2 only: app/extraction/regex_hints.py and app/extraction/grounding.py.

Tasks:
1. Regex hints from native text layers: GSTINs (checksum-valid only, plus a separate list of
   checksum-invalid candidates), PANs, emails, phones, dates (common Indian formats → ISO), HSN/SAC-like codes.
2. GroundingIndex built from native pages: normalised corpus (uppercase, spaces and separators removed).
   is_grounded(value) → "yes" | "no" | "unverifiable":
   found → yes; not found and the document is fully native → no;
   not found and the document has any scanned page → unverifiable (never a failure on its own).
3. Hints are used for validation and grounding only — they are NOT injected into the prompt (keeps the
   prompt prefix stable and avoids biasing the model). Document this choice in a docstring.
4. Tests on data/synthetic native docs: all truth GSTINs/PANs found; grounding "yes" for truth identifiers,
   "no" for a fabricated GSTIN on a native doc, "unverifiable" on a scanned twin.

Finish with a summary and verification commands.
```

**Verify:** `make test`

---

## Step 3.3 — Message builder (prefix-stable) [plan]

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 6.3, 9.2 and 9.3. Read schema/llm_keys.py,
schema/llm_schemas.py and app/extraction/preprocess.py.

Implement Step 3.3 only: app/extraction/prompts.py.

Tasks:
1. System prompt (constant text, versioned with PROMPT_VERSION): role = precise document data extractor;
   rules: extract only values printed on the document; never calculate; omit fields that are absent;
   dates as YYYY-MM-DD; numbers plain (no thousands separators, Indian "1,23,456.50" → 123456.50);
   copy identifiers (GSTIN, PAN, codes) exactly; common label synonyms (Supplier = vendor,
   Consignee/Deliver To = ship-to, Invoice To = bill-to, Basic Amount = taxable value).
2. Page content block (shared prefix): for each non-skipped page in order:
   "Page N of M (native|scanned)", then the image part, then (native only, per native_mode) the text layer
   in a clearly delimited block. Identical bytes for every call of the same document.
3. Task blocks (appended last, differ per call):
   - header task: field definitions generated from header_schema_for_llm() descriptions (short keys).
   - line-items task (all pages): column order from line_item_columns_instruction(); one row per printed line
     item; rows split across a page break are ONE row; output {"rows": [...]}.
   - line-items task (single page N): only rows whose row number first appears on page N; a row continuing
     from the previous page is not repeated.
   - Hook: supplier_hints(issuer_id) → "" for now (phase-2 feature placeholder), placed inside the task block.
4. build_messages(prepared_doc, task) → OpenAI-format messages (image parts as data URLs).
5. Tests: for one document, the header, all-lines and per-page tasks produce messages whose everything
   except the final task block is byte-identical; skipped pages absent; scanned pages have no text block;
   prompt contains no computed-field names.

Finish with a summary and verification commands.
```

**Verify:** `make test`; print one header-call message (images replaced by placeholders) and read it yourself — it should look like something you'd hand a careful human.

---

## Step 3.4 — LLM client with request profiles

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 7 (FR-08, FR-13) and 9.5. Read schema/ and
infra/litellm/. Add the openai Python package if not present (used only as a client for LiteLLM).

Implement Step 3.4 only: app/extraction/llm_client.py and request profiles.

Tasks:
1. config/llm_profiles.yaml with two profile sets selected by env LLM_PROFILE_SET = dev | h100.
   Each alias (po-fast, po-accurate, po-baseline, po-moe) defines its request parameters:
   - h100: po-fast → temperature 0, presence_penalty 0, repetition_penalty 1.0,
     chat_template_kwargs {enable_thinking: false}, send_priority: true.
     po-accurate → same but thinking on with low reasoning effort (put the exact parameter shape in ONE place,
     marked "VERIFY against the Qwen3.8 model card / vLLM version in Phase 7").
   - dev (llama.cpp): po-fast thinking off; po-accurate thinking on (llama.cpp has no effort levels);
     send_priority: false (llama.cpp does not support vLLM priority).
   Settings loads the active profile set; nothing model-specific is hard-coded in Python.
2. LlmClient.call(alias, messages, schema, max_tokens, priority=None, call_id) → CallResult:
   parsed JSON (short keys mapped to full names via schema/llm_keys.py), raw content, reasoning content
   (if any, kept only for debugging), tokens {prompt, completion, reasoning if reported},
   timings {start_ts, end_ts, duration_ms}, error.
   - response_format json_schema; max_tokens safety cap from settings (header 2000, line items 6000 by default).
   - Per-call timeout (default 120 s). Transient errors (timeout, connection, 429, 5xx) retried with
     exponential backoff (max 2) — separate from validation retries in Step 3.7.
   - Invalid/truncated JSON → error result, not exception.
3. Tests with a mocked HTTP transport: correct request body per profile (thinking flags, penalties,
   priority only when send_priority), short→full key mapping, transient retry then success, truncation handling.

Finish with a summary and verification commands.
```

**Verify:** `make test`; quick live call: a tiny script that runs a header call on one native dev PO through LiteLLM and prints tokens and duration.

---

## Step 3.5 — Call planner and orchestrator [plan]

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 7 (FR-07, FR-14, FR-15) and 9.3. Read
app/extraction/prompts.py and llm_client.py.

Implement Step 3.5 only: app/extraction/planner.py and app/extraction/orchestrator.py.

Tasks:
1. Planner: plan_calls(prepared_doc, strategy) → list of CallSpec(call_id, task, alias, max_tokens).
   - two_call: H + LI(all pages).
   - per_page: H + LI-p{n} for each non-skipped page likely to contain line items (native: page markdown
     contains table rows; scanned: every non-skipped page).
   - adaptive (default): two_call, unless estimated rows > ADAPTIVE_ROW_THRESHOLD (default 40) or item-bearing
     pages > ADAPTIVE_PAGE_THRESHOLD (default 2) → per_page for line items. Row estimate for native pages:
     count table rows starting with a line number; for scanned pages use page count only.
   - Record the decision and its reason in the result (routing.strategy, routing.reason).
2. Orchestrator: run_calls(doc, specs, po_priority) → results.
   - All calls of a PO run concurrently (asyncio.gather).
   - A process-wide asyncio.Semaphore limits in-flight LLM calls to settings.llm_concurrency
     (must stay ≤ vLLM max-num-seqs; document this).
   - po_priority = PO arrival sequence number (smaller = earlier = served first); passed to the client.
   - Timeline per call (start/end relative to PO start) for the dashboard Gantt view.
   - One failed call never cancels the others; failures are returned for Step 3.7.
3. Tests with a fake client that sleeps: two_call and per_page calls overlap in time (total ≈ slowest call);
   semaphore caps concurrency; adaptive switches at the thresholds; decision reason recorded.

Finish with a summary and verification commands.
```

**Verify:** `make test`

---

## Step 3.6 — Merge, computed fields and validation engine [plan]

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 6.3 and 9.4 (all 10 rules). Read schema/,
datagen/consistency.py (reuse logic where it fits; do not duplicate GSTIN/rounding helpers),
app/extraction/grounding.py and data/synthetic/masters/.

Implement Step 3.6 only: app/extraction/merge.py, app/extraction/compute.py, app/extraction/validate.py.

Tasks:
1. Merge: combine header + line-item results into a PurchaseOrder. Per-page mode: concatenate rows in page
   order, deduplicate by line_no (keep the most complete row), repair continuation fragments (a row without
   line_no/quantity at the top of a page whose description continues the previous row → append to that row).
2. Compute COMPUTED_FIELDS exactly as the generator does (same rounding: ROUND_HALF_UP, 2 dp):
   supply type intra/inter from vendor state code vs place-of-supply state code; per-line CGST/SGST or IGST;
   total_tax.
3. Validation engine: one function per PRD §9.4 rule, each returning CheckResult(rule, passed, fields,
   detail, retry_target: "H" | "LI" | "LI-p{n}" | "both" | None, retryable: bool).
   - Header-only failures → retry H. Row/continuity failures → retry the LI call that produced those rows.
     Totals-vs-lines mismatch → both. Master-data duplicate PO number → not retryable (goes straight to review).
   - Grounding uses GroundingIndex: "no" fails, "unverifiable" passes with a note.
   - Masters loaded once from a configurable path (default data/synthetic/masters/).
4. Tests: one test per rule (pass and fail); ALL truth files in data/synthetic pass every rule except the
   expected duplicates, which fail only the duplicate rule (and are marked not retryable).

Finish with a summary and verification commands.
```

**Verify:** `make test`

---

## Step 3.7 — Targeted retry and review routing

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 3, 6.4 and 7 (FR-11, FR-12). Read app/extraction/.

Implement Step 3.7 only: app/extraction/pipeline.py — the end-to-end extract(pdf_bytes) function.

Flow: prepare → plan → run calls (po-fast) → merge → compute → validate →
  all pass → status completed
  retryable failures → re-run ONLY the failed calls once on po-accurate → merge → compute → re-validate
    → pass → completed (routing.retried_calls recorded) / fail → needs_review
  non-retryable failures → needs_review without retry
  call errors after transient retries → failed (job layer handles DLQ in Phase 4)

Requirements:
1. Retry deadline: skip the retry if elapsed time already exceeds RETRY_DEADLINE_S (default 40) → needs_review,
   so a PO never exceeds the 60 s ceiling because of a retry. Record the reason.
2. needs_review results list failed rules and failed fields (for highlighting in the dashboard).
3. Return the full result envelope from PRD §6.4 (status, data, validation, routing, pages, timings, tokens,
   call timeline).
4. Tests with a fake client: injected wrong GSTIN → only H retried on po-accurate; dropped row on page 2 in
   per-page mode → only LI-p2 retried; duplicate PO → needs_review, no retry; deadline exceeded → no retry.

Finish with a summary and verification commands.
```

**Verify:** `make test`

---

## Step 3.8 — Optimised pipeline CLI + first comparison

```
Read CLAUDE.md, docs/PRD.md (v1.1) sections 11 and 12, eval/run_format.py and app/extraction/pipeline.py.

Implement Step 3.8 only.

Tasks:
1. app/extraction/run.py CLI:
   `uv run python -m app.extraction.run --split dev --variants N,S,M --strategy adaptive --run-id opt_dev_local
    --concurrency 4`
   Options for ablations: --strategy two_call|per_page|adaptive, --dpi, --native-mode text+image|text|image,
   --no-skip-pages, --no-retry. Documents are submitted concurrently up to --concurrency, each with an
   increasing po_priority. Writes the standard run format (outputs in FULL field names, raw calls,
   records with timings, tokens, routing, status).
2. Makefile: `make extract SPLIT=dev VARIANTS=N,S,M STRATEGY=adaptive RUN=opt_dev_local`.
3. README section "Running the optimised pipeline" incl. the model restart command for 4 parallel slots.
4. A smoke test that runs the CLI on 1 document with a fake client and produces a valid run folder.

Finish with a summary and the exact commands to run and compare against the baseline.
```

**Verify:**
1. `PARALLEL=4 CTX_SIZE=65536 make model` (restart), `make litellm`
2. `make extract SPLIT=dev VARIANTS=N,S,M STRATEGY=adaptive RUN=opt_dev_local`
3. `make eval RUN=runs/opt_dev_local`
4. `make compare REF=runs/b0_dev_local CAND=runs/opt_dev_local` (compares the docs both runs share)

Expect locally: far fewer completion tokens and much lower latency than the baseline. Absolute accuracy on
the 9B model will be lower than on the 27B — what matters now is that the pipeline works end to end and the
review/retry paths behave correctly. Phase 3 complete — commit and tag `phase-3`.
