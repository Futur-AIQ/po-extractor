# Phase 2 — Baseline & evaluation harness (Claude Code prompts) — v1.1

Goal: (1) a scoring engine that can judge any run against ground truth, split by native / scanned / mixed,
layout and issuer type; (2) a baseline that faithfully reproduces today's setup; (3) a CLI that reports,
compares runs, and applies the accuracy acceptance rule (PRD §11.3).

Order: 2.1 → 2.2 → 2.3. Run `/clear` before each prompt. Commit after each step.

---

## Step 2.1 — Run format, normalisation and scoring [plan]

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 3, 6, 11.2 and 11.3. Read schema/ and the dataset layout
produced in Phase 1 (data/synthetic: pdfs/, truth/, masters/, manifest.csv, splits.json; variants N/S/M).
Add rapidfuzz as a dependency.

Implement Step 2.1 only: a shared run format and the scoring engine in eval/.

Tasks:
1. eval/run_format.py — the standard on-disk format every pipeline (baseline, optimised, benchmark) writes:
   runs/<run_id>/config.json            (pipeline name, model alias, strategy, dpi, thinking, git commit, timestamp)
   runs/<run_id>/outputs/<doc_id>.json  (final PurchaseOrder in FULL field names, or an error record)
   runs/<run_id>/raw/<doc_id>.json      (raw model responses, for debugging)
   runs/<run_id>/records.jsonl          (one line per doc: doc_id, variant, status, timings_ms,
                                         tokens {prompt, completion, reasoning if known}, calls, error)
   Provide typed writer/reader helpers. doc_id includes the variant suffix (PO_0001, PO_0001_S, PO_0001_M).
2. eval/normalize.py — per field-type normalisation:
   identifiers (GSTIN, PAN, PO number, item codes, HSN): uppercase, remove spaces;
   numbers: Decimal, compared to 2 decimal places; dates: ISO; emails: lowercase;
   phones: digits only, last 10; free text (names, addresses, descriptions, terms): casefold,
   collapse whitespace, strip trailing punctuation.
   A field-type registry maps every schema field to its type (test: every field mapped).
3. eval/score.py:
   - Header fields: for each field classify as correct, wrong, missed (truth present, pred absent),
     hallucinated (truth absent, pred present), or correct-absent. Free-text fields also get a fuzzy
     score (rapidfuzz token_set_ratio); report exact and fuzzy (≥ 90) accuracy separately.
   - Line items: match rows by line_no; for rows without a usable line_no, fall back to greedy matching on
     item_code, quantity and fuzzy description. Report row recall, row precision, and per-column
     cell accuracy over matched rows.
   - Critical-field document accuracy: a doc counts as correct only if every critical header field is
     correct, every matched row's critical columns are correct, and row recall is 100%.
   - Option to include or exclude COMPUTED_FIELDS (default: include — final output is what matters).
   - Aggregations: overall and by variant (N/S/M), layout, issuer_frequent.
   - Error list: one record per wrong/missed/hallucinated value (doc_id, variant, field, truth, pred).
   - Latency and tokens summary from records.jsonl: p50/p95/max total latency; mean prompt, completion
     and reasoning tokens; parse-error count.
4. Tests (no LLM): truth scored against itself = 100% everywhere; a hand-made prediction with one wrong
   GSTIN, one missed field, one hallucinated field and one dropped row produces exactly those counts;
   Decimal comparison ignores formatting ("1,23,456.50" vs 123456.5); variant aggregation correct.
   Add a helper scripts/make_oracle_run.py that writes a run where outputs = truth (used to test reports).

Do not build the CLI or any pipeline yet.
Finish with a summary and verification commands.
```

**Verify:** `make test`; `uv run python scripts/make_oracle_run.py --split dev` creates `runs/oracle_dev/`.

---

## Step 2.2 — Baseline pipeline B0 (reproduces today's setup) [plan]

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 1, 9.5 and 12 (row B0). Read schema/, eval/run_format.py
and infra/llamacpp/start.sh.

Implement Step 2.2 only: baseline/ — a faithful reproduction of the CURRENT production approach, so later
improvements are measured against reality, not a straw man.

Baseline behaviour (configurable, defaults reproduce today):
- All pages of the PDF rendered with PyMuPDF at 120 dpi, sent as images in ONE request. No text layer.
- Full JSON schema with FULL field names, all fields including computed ones, line items as objects
  (what a typical guided-JSON setup looks like today).
- Model alias po-baseline. Thinking at the model's default (do not send enable_thinking).
  No temperature override by default (flag --temperature to set one).
- Large max_tokens (default 16000) and a long timeout (default 600 s).

Tasks:
1. baseline/pipeline.py: async extract_baseline(pdf_path, settings) → (PurchaseOrder | error, raw response,
   timings {render, llm, parse, total}, tokens). Read reasoning token counts if the response reports them
   (usage.completion_tokens_details or reasoning_content length); store reasoning text in raw only.
   Malformed or truncated JSON → recorded as a parse error, never a crash.
2. baseline/run.py CLI:
   `uv run python -m baseline.run --split dev --variants N,S --limit 3 --concurrency 1 --run-id b0_dev_local`
   Writes the standard run format from Step 2.1. Shows a progress line per doc (doc_id, seconds,
   completion tokens).
3. Local model settings for long multimodal requests: update infra/llamacpp/start.sh so PARALLEL and
   CTX_SIZE come from env vars (defaults 4 and 32768). Explain in README that llama.cpp divides the context
   across parallel slots, so baseline runs need e.g. PARALLEL=1 CTX_SIZE=32768 (images + thinking output
   exceed 8k per slot). Verify the vision projector (mmproj) is loaded at startup and document how to
   check it; if missing, document the fix rather than guessing flags (check llama-server --help).
4. Makefile: `make baseline SPLIT=dev VARIANTS=N,S LIMIT=3`.
5. Tests: mocked LiteLLM response → correct run files; truncated JSON → parse-error record;
   page rendering produces one image per page at the requested DPI.

Do not implement the optimised pipeline or evaluation CLI.
Finish with a summary, the exact commands to restart the local model for the baseline, and verification commands.
```

**Verify:**
1. Restart the local model for long requests: `PARALLEL=1 CTX_SIZE=32768 make model`
2. `make baseline SPLIT=dev VARIANTS=N,S LIMIT=3`
3. Check `runs/<run_id>/records.jsonl`: completion tokens should be large if thinking is on — this is the
   local version of the root-cause evidence.

Note: on the Mac, the 9B model with thinking on can take minutes per PO. Use `LIMIT=3` locally; the full
baseline runs on the H100 in Phase 7.

---

## Step 2.3 — Evaluation CLI, reports and run comparison

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 3, 11.2 and 11.3. Read eval/.

Implement Step 2.3 only.

Tasks:
1. eval/report.py + CLI `uv run python -m eval.report --run runs/<run_id>`:
   - report.md and report.json inside the run folder: overall accuracy table, breakdown by variant
     (N / S / M), layout and issuer type, line-item metrics, critical-field document accuracy,
     latency p50/p95/max, mean tokens, parse errors.
   - errors.csv: all error records, sorted by field frequency.
   - "Top 10 problem fields" section with 2 examples each (truth vs prediction).
   - Prints a compact summary to the terminal.
2. eval/compare.py + CLI `uv run python -m eval.compare --ref runs/<A> --cand runs/<B>`:
   - Compares only docs present in both runs.
   - Side-by-side table: accuracy metrics, latency p50/p95, mean completion tokens, with deltas.
   - Applies the PRD §11.3 acceptance rule (critical-doc accuracy drop ≤ 0.5 pp AND field accuracy
     drop ≤ 1 pp) and prints ACCEPT or REJECT with the reason. Writes compare.md.
3. Makefile: `make eval RUN=runs/<id>` and `make compare REF=runs/<A> CAND=runs/<B>`.
4. Tests: oracle run → 100% report; compare oracle vs a degraded copy → REJECT with the correct reason;
   compare oracle vs itself → ACCEPT with zero deltas.

Finish with a summary and verification commands.
```

**Verify:**
1. `make eval RUN=runs/oracle_dev` → 100% everywhere.
2. `make eval RUN=runs/b0_dev_local` → first real baseline report; read the top problem fields.
3. `make compare REF=runs/oracle_dev CAND=runs/b0_dev_local` → shows the gap to perfect.

Phase 2 complete — commit and tag `phase-2`.
