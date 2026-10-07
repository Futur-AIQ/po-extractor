# Phase 1 — Schema & synthetic data (Claude Code prompts) — v1.1

> Steps 1.1–1.4 are complete. Continue with **1.4a → 1.4b → 1.5 → 1.6** below
> (these reflect PRD v1.1: computed fields, short keys, issuer pool, ERP masters, scanned twins).

Goal: a realistic, varied, internally consistent set of Indian GST purchase orders (2–3 page PDFs)
with exact ground-truth JSON. This dataset is the foundation for evaluation and benchmarking.

Run `/clear` before each prompt. Commit after each step. Steps marked [plan] → use plan mode first.

---

## Step 1.1 — PO schema (single source of truth) [plan]

```
Read CLAUDE.md and docs/PRD.md sections 6, 9.2 and 9.3.

Implement Step 1.1 only: the Pydantic schema in schema/.

Tasks:
1. schema/po_schema.py with Pydantic v2 models:
   - LineItem: the 15 line-item fields from PRD 6.2.
   - PurchaseOrder: the 58 header fields from PRD 6.1 plus line_items: list[LineItem].
   - Money and quantity fields use Decimal; dates use datetime.date; all non-critical fields Optional.
   - Field descriptions on every field (short, precise; they become instructions inside the LLM JSON schema).
     Include common label synonyms in descriptions, e.g. ship_to may appear as "Consignee" or "Deliver To";
     vendor as "Supplier"; po_number as "Order No.", "P.O. Ref".
2. schema/fields.py:
   - CRITICAL_FIELDS: set of critical header and line-item field names (marked * in the PRD).
   - FIELD_GROUPS mapping group id → list of header fields:
       G1_HEADER_TERMS = document (10) + terms (8) + approval (2)
       G2_PARTIES      = buyer (9) + vendor (10) + bill-to (4) + ship-to (4)
       G3_TOTALS       = totals (11)
     A test must prove every header field belongs to exactly one group.
   - LINE_ITEM_COLUMNS: ordered list of line-item fields for the compact format.
3. schema/llm_schemas.py:
   - json_schema_for_group(group_id) → JSON schema dict for that group's fields only, for use in
     response_format json_schema. Numbers as JSON numbers, dates as "YYYY-MM-DD" strings, all fields
     optional (the model omits absent fields; no nulls required).
   - compact_line_items_schema() → schema for {"rows": [[...], ...]} where each row is an array in
     LINE_ITEM_COLUMNS order (column order is given in the prompt, not repeated per row).
   - rows_to_line_items(rows) → list[LineItem] converter with clear errors on malformed rows.
   Keep schemas simple and flat; vLLM and llama.cpp structured output must both accept them
   (avoid unusual JSON-schema keywords; no $ref if avoidable).
4. Tests: group coverage, schema generation for each group, compact row round-trip,
   Decimal precision preserved (no float conversion).

Do not implement validation rules (Step 3.6) or any LLM calls.
Finish with a summary and verification commands.
```

**Verify:** `make test`; print one group schema: `uv run python -c "from schema.llm_schemas import json_schema_for_group as s; import json; print(json.dumps(s('G3_TOTALS'), indent=2))"`

---

## Step 1.2 — Indian business data providers

```
Read CLAUDE.md and docs/PRD.md section 11.1. Read schema/ from Step 1.1.

Implement Step 1.2 only: datagen/india.py — realistic Indian business data, fully seeded (random.Random instance
passed in, never global random).

Tasks:
1. STATES: all Indian states and UTs with official GST state codes (e.g. 24 Gujarat, 27 Maharashtra,
   29 Karnataka, 33 Tamil Nadu, 07 Delhi). For ~10 major industrial states include 4–6 real cities each
   with a plausible 6-digit PIN code prefix for that state.
2. GSTIN generator with the official mod-36 checksum: state code (2) + PAN (10) + entity number (1)
   + 'Z' + checksum (1). Also is_valid_gstin(gstin) and pan_from_gstin(gstin).
   Tests: generated GSTINs validate; a known-valid sample validates; a one-character change fails.
3. PAN generator (company PANs: 4th character 'C').
4. Company name generator (Pvt. Ltd., Ltd., LLP, industries, engineering, chemicals, etc.) and
   addresses consistent with the chosen state (plot/survey numbers, GIDC/MIDC industrial estates,
   city, PIN). Use Faker en_IN only for person names, phones and emails.
5. Item catalogue: ~120 realistic items across 6 industries (industrial equipment, chemicals, electrical,
   fasteners/hardware, IT/office, services) with item code pattern, description (some long, multi-line
   capable), HSN code (goods) or SAC code (services), typical UoM (NOS, KG, MTR, LTR, SET, BOX, HRS),
   price range, and GST rate.
6. GST_SLABS configurable in one constant with a comment to verify the current official rates;
   default to 0, 5, 18 and 40 percent with realistic weights (most items 18%).
7. amount_in_words_inr(Decimal) → Indian numbering system ("Rupees Two Lakh Forty Five Thousand
   Three Hundred Twelve and Fifty Paise Only"), using num2words where it helps. Tests for lakh/crore/paise.
8. Payment, delivery, freight terms, transport modes, warranty and packing text pools.

Do not build the PO generator yet.
Finish with a summary and verification commands.
```

**Verify:** `make test`

---

## Step 1.3 — PO data generator with difficulty knobs [plan]

```
Read CLAUDE.md, docs/PRD.md sections 6, 9.4 and 11.1. Read schema/ and datagen/india.py.

Implement Step 1.3 only: datagen/generate.py producing PurchaseOrder objects (ground truth).

Tasks:
1. Knobs dataclass (all seeded, all overridable): line count range 30–60, probability of
   inter-state supply (~40%), multi-line description, discount per line, freight charges, other charges,
   amendment fields present, quotation/indent refs present, optional-field drop rate (~15% of optional
   fields absent), bill-to different from buyer, ship-to different from bill-to, T&C page present (~50%).
2. Business consistency (this is the ground truth, so it must be exact):
   - buyer/vendor GSTIN state codes match their addresses; PAN matches GSTIN.
   - place_of_supply = ship-to state; intra-state → CGST+SGST (each half the rate, per line), IGST 0;
     inter-state → IGST only.
   - taxable_value = qty × rate × (1 − discount%) rounded to 2 decimals (ROUND_HALF_UP);
     line taxes rounded to 2 decimals; line_total = taxable + line taxes.
   - subtotal = Σ taxable; tax totals = Σ line taxes; grand_total rounded to nearest rupee with round_off
     (|round_off| ≤ 0.50); amount_in_words matches grand_total.
   - delivery_date ≥ po_date; line_no sequential from 1.
   - PO number formats vary by buyer (e.g. PO/2026-27/00457, 4500012345, GMM-PUR-26-1182).
3. datagen/consistency.py: check_po(po) → list of violations implementing the arithmetic/tax/GSTIN checks above.
   (The production validation engine comes in Step 3.6; this is the generator's own self-check.)
4. generate_po(seed, knobs) and generate_many(n, seed, knobs).
5. Tests: 200 generated POs have zero violations; same seed → identical output; intra/inter-state mix
   roughly matches the knob; JSON serialisation round-trips with Decimal precision.

Do not render PDFs yet.
Finish with a summary and verification commands.
```

**Verify:** `make test`

---

## Step 1.4 — Six PO layouts [plan]

```
Read CLAUDE.md and docs/PRD.md section 11.1. Read schema/ and datagen/generate.py.

Implement Step 1.4 only: six Jinja2 HTML templates in datagen/templates/ that render a PurchaseOrder
as A4 print-ready HTML. The goal is genuine layout diversity, like POs from different companies' ERPs.

Layouts (each must look clearly different):
- L1_classic_erp: boxed header grid, buyer and vendor side by side, full per-line tax columns.
- L2_tally_style: dense bordered table with CGST/SGST/IGST columns, plus an HSN-wise tax summary table
  after the items (repeats numbers — a realistic distraction).
- L3_modern_minimal: no table borders, key-value header list, vendor block on the right, light colours.
- L4_engineering: item code + drawing/spec reference, long multi-line descriptions, per-line delivery dates.
- L5_label_variants: uses synonym labels ("Order Ref", "Supplier", "Consignee", "Invoice To",
  "Rate/Unit", "Basic Amount") to test label robustness.
- L6_psu_formal: formal letter-style opening paragraph, items table, totals, then a terms & conditions
  page with numbered clauses.

Requirements:
1. Shared base template + per-layout templates; CSS for print: @page A4 with margins, page numbers
   "Page X of Y" in footer, repeated company header on every page, items table with
   thead { display: table-header-group } so headers repeat on page breaks, rows allowed to break across pages
   only when the knob says so (otherwise page-break-inside: avoid on rows).
2. Number formatting in Indian style (1,23,456.78); date formats vary by layout (DD-MM-YYYY, DD/MM/YYYY,
   DD-Mon-YYYY) — the ground truth stays ISO.
3. Optional fields that are absent in the PO must not render at all (no empty labels).
4. Render the T&C page only when the knob is set (L6 always has it).
5. datagen/templates/__init__.py: render_html(po, layout_id) → str.
6. A preview script scripts/preview_layouts.py that renders one PO in all six layouts to
   data/preview/*.html for visual review.

Do not render PDFs yet.
Finish with a summary and how to open the previews.
```

**Verify:** `uv run python scripts/preview_layouts.py`, open the six HTML files in a browser, check they look distinct and realistic.

---

## Step 1.4a — Schema patch: computed fields, short keys, LLM schemas

```
Read CLAUDE.md, docs/PRD.md (v1.1) sections 6.3 and 9.3. Read the existing schema/ package.

Implement Step 1.4a only: extend the schema for the v1.1 LLM output contract. Do not break existing
APIs or tests from Step 1.1; extend them.

Tasks:
1. schema/fields.py additions:
   - COMPUTED_FIELDS: header {"total_tax"} and line items {"cgst_amount", "sgst_amount", "igst_amount"}.
     These are never requested from the model; they are computed in code later (Step 3.6).
   - HEADER_FIELDS_FOR_LLM: all 58 header fields minus computed ones.
   - LLM_LINE_ITEM_COLUMNS: LINE_ITEM_COLUMNS minus computed ones, same relative order.
2. schema/llm_keys.py:
   - SHORT_KEYS: full field name → short key (max 8 chars, unique, still readable, e.g. po_no, po_dt,
     v_gstin, b_gstin, st_gstin, gr_total). Reverse map built automatically. Test: unique and complete.
   - to_full_keys(dict) and to_short_keys(dict).
3. schema/llm_schemas.py additions (keep existing functions):
   - header_schema_for_llm(): one flat JSON schema with ALL header fields for the LLM (short keys,
     descriptions kept meaningful and including label synonyms), all optional, absent fields omitted.
   - line_items_schema_for_llm(): {"rows": [[...]]} using LLM_LINE_ITEM_COLUMNS.
   - line_item_columns_instruction(): a short text listing the column order with one-line meanings
     (used later inside the prompt instead of repeating keys in every row).
   - rows_to_line_items(rows, columns=LLM_LINE_ITEM_COLUMNS) works with the reduced column list;
     computed columns are left None for later computation.
4. Tests: computed fields absent from both LLM schemas; short-key round trip on a full PurchaseOrder;
   reduced row round trip; schemas use only simple JSON-schema keywords accepted by vLLM and llama.cpp.

Finish with a summary and verification commands.
```

**Verify:** `make test`

---

## Step 1.4b — Generator patch: issuer pool, line counts, mock ERP masters [plan]

```
Read CLAUDE.md, docs/PRD.md (v1.1) sections 5, 9.4 (rule 10) and 11.1. Read datagen/.

Implement Step 1.4b only: make the generated data reflect the confirmed real-world distribution.
Keep all existing consistency tests passing.

Tasks:
1. datagen/issuers.py:
   - A fixed, seeded pool of 30 FREQUENT issuers (the company whose ERP produced the PO). Each has:
     id, company profile (name, GSTIN, PAN, address, state), an assigned layout (5 issuers per layout,
     L1–L6), its own PO-number format, and its own date format preference.
     The pool must be identical across runs (own fixed seed), independent of the dataset seed.
   - A fixed, seeded pool of ~40 counterparties (the other party on the PO).
   - make_one_off_issuer(rng): a random issuer with a random layout.
2. Knobs additions: frequent_issuer_share (default 0.70); line count drawn from 20–80 with most POs
   in 30–50; duplicate_po_count (default 2): POs whose number is deliberately pre-registered as
   "already processed" in the masters.
3. generate_po() uses the issuer's profile, layout, PO-number format and date format. If templates
   already fix date formats per layout, keep that and store the issuer's format only as metadata.
4. datagen/masters.py: build_masters(pos, issuers, counterparties, catalogue) →
   parties.json (name, GSTIN, state code for every party appearing in the dataset),
   items.json (all item codes in the catalogue), processed_po_numbers.json (a few hundred plausible
   historical PO numbers per frequent issuer + the deliberate duplicates).
   Truth meta gains: issuer_id, issuer_frequent (bool), expected_duplicate (bool).
5. Tests: issuer pool stable across runs; frequent share ≈ 70% over 500 POs; every party and item in
   generated POs exists in the masters; exactly duplicate_po_count POs are expected duplicates;
   all existing consistency tests still pass.

Do not render PDFs (Step 1.5).
Finish with a summary and verification commands.
```

**Verify:** `make test`

---

## Step 1.5 — PDF rendering + dataset build CLI (v1.1)

```
Read CLAUDE.md and docs/PRD.md (v1.1) section 11.1. Read datagen/.

Implement Step 1.5 only.

Tasks:
1. datagen/render.py: async Playwright (Chromium) renderer — one browser, several pages concurrently —
   HTML → A4 PDF with backgrounds. Return PDF bytes.
2. Page-count control with PyMuPDF: target 2–4 pages; 5 allowed only when a T&C page is present.
   If outside range, adjust the line count deterministically (same seed) and regenerate, max 5 attempts,
   then log and skip.
3. Per-page truth metadata via PyMuPDF text search: which line_no values appear on each page, and which
   pages contain no extractable fields (e.g. T&C pages).
4. datagen/build.py CLI: `uv run python -m datagen.build --n 60 --seed 42 --out data/synthetic`
   - layout comes from the issuer (frequent issuers fixed; one-off issuers random); ensure each layout
     appears at least 6 times (rebalance one-off issuers if needed)
   - writes pdfs/PO_0001.pdf, truth/PO_0001.json (PurchaseOrder + meta: issuer_id, issuer_frequent,
     expected_duplicate, layout, knobs, seed, page_count, lines_per_page, fieldless_pages),
     masters/ (from Step 1.4b), manifest.csv
   - splits.json: dev 10 POs (at least one per layout) and test 50, stratified by layout
   - preview.html: thumbnail grid of page 1 of every PO for visual QA
5. Makefile: `make dataset` (runs `uv run playwright install chromium` if needed).
6. Tests: build 3 POs into a temp dir; PDFs open; page counts in range; truth passes
   datagen/consistency.py; extracted PDF text contains the PO number and the grand total in Indian format.

Finish with a summary and verification commands.
```

**Verify:** `make dataset`, open `data/synthetic/preview.html`, compare 2–3 PDFs with their truth JSON.

---

## Step 1.6 — Scanned twins, mixed POs, public datasets (v1.1)

```
Read CLAUDE.md and docs/PRD.md (v1.1) sections 5 and 11.1. Read datagen/.
Add pillow and numpy if not present.

Implement Step 1.6 only.

Tasks:
1. datagen/scanify.py: convert a native PDF page into a realistic scanned page: rasterise at 200 dpi
   (typical scanner resolution), slight random rotation (±1°), mild noise and blur, JPEG compression,
   and rebuild as image-only PDF pages (no text layer). Seeded.
2. Scanned twins: for EVERY PO in data/synthetic, create data/synthetic/pdfs/PO_0001_S.pdf with the same
   truth (truth file references the twin; meta page kinds all "scanned").
3. Mixed POs: for 10 POs (spread across layouts, from the test split), create PO_00xx_M.pdf where 1–2
   random pages are scanned and the rest stay native; record page kinds in the truth meta.
4. Update manifest.csv and splits.json so each split lists native, scanned twin and mixed variants
   (variant column: N / S / M). Benchmark mixes can then sample 50/50 native-scanned.
5. CLI: `uv run python -m datagen.scanify --src data/synthetic` (idempotent); Makefile `make scanned`.
6. scripts/download_public.py (idempotent):
   - Northwind: huggingface_hub snapshot_download of dataset AyoubChLin/northwind_PurchaseOrders → data/northwind/
   - FATURA: stream https://zenodo.org/records/10371464/files/FATURA2.zip?download=1 to data/fatura/ with
     progress, verify MD5 4c9404462f22c5241eb1a290a02eb2a2, unzip, copy a 50-image sample (one per template,
     white background) with annotations to data/fatura/sample_50/
   - README credits: FATURA (CC BY 4.0), Northwind dataset.
   Makefile `make public-data`.
7. Tests: twin PDFs have no extractable text and the same page count; mixed PDFs have the recorded
   page kinds (text present only on native pages).

Finish with a summary and verification commands.
```

**Verify:** `make scanned && make public-data`. Phase 1 complete — commit and tag `phase-1`.
