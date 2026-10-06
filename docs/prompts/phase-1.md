# Phase 1 — Schema & synthetic data (Claude Code prompts)

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

## Step 1.5 — PDF rendering + dataset build CLI

```
Read CLAUDE.md and docs/PRD.md section 11.1. Read datagen/.

Implement Step 1.5 only.

Tasks:
1. datagen/render.py: async Playwright (Chromium) renderer — one browser, multiple pages concurrently —
   HTML → A4 PDF with backgrounds printed. Return PDF bytes.
2. Page-count control: after rendering, count pages with PyMuPDF. Target 2–3 pages
   (allow 4 only for layouts with a T&C page). If outside range, adjust the line count and regenerate
   (deterministically, from the same seed) — max 5 attempts, then log and skip.
3. Record per-page line mapping: using PyMuPDF text search, store which line_no values appear on
   each page (truth metadata used later for per-page evaluation and debugging).
4. datagen/build.py CLI: `uv run python -m datagen.build --n 60 --seed 42 --out data/synthetic`
   - layouts assigned round-robin (10 per layout)
   - writes pdfs/PO_0001.pdf, truth/PO_0001.json (PurchaseOrder + meta: layout, knobs, seed,
     page_count, lines_per_page), manifest.csv
   - splits.json: dev (10 POs, at least one per layout) and test (50), stratified by layout
   - preview.html: thumbnail grid of page 1 of every PO (PyMuPDF render) for quick visual QA
5. Makefile: `make dataset` (also runs `uv run playwright install chromium` if needed).
6. Tests: build 3 POs into a temp dir; PDFs open; page counts in range; truth validates with
   datagen/consistency.py; text extracted from each PDF contains the PO number and grand total
   (in Indian format) — proves the truth matches what is printed.

Finish with a summary and verification commands.
```

**Verify:** `make dataset`, open `data/synthetic/preview.html`, open 2–3 PDFs and compare against their truth JSON.

---

## Step 1.6 — Scanned variants + public dataset downloads

```
Read CLAUDE.md and docs/PRD.md section 11.1. Read datagen/.

Implement Step 1.6 only. Add pillow and numpy as dependencies if not present.

Tasks:
1. datagen/scanify.py: convert a digital PDF into a realistic image-only "scanned" PDF:
   render each page at 150 DPI, slight random rotation (±1°), mild noise, slight blur, JPEG compression,
   then rebuild a PDF of images (no text layer). Seeded.
   CLI: `uv run python -m datagen.scanify --src data/synthetic --n 10 --out data/synthetic_scanned`
   picks 10 POs from the test split (spread across layouts) and copies their truth files.
   Test: output PDF has no extractable text; page count unchanged.
2. scripts/download_public.py:
   - Northwind: huggingface_hub snapshot_download of dataset AyoubChLin/northwind_PurchaseOrders
     into data/northwind/.
   - FATURA: stream-download https://zenodo.org/records/10371464/files/FATURA2.zip?download=1
     to data/fatura/ with a progress bar, verify MD5 4c9404462f22c5241eb1a290a02eb2a2, unzip,
     then copy a sample of 50 images (one per template, white background) plus their annotation files
     into data/fatura/sample_50/. Skip steps already completed (idempotent).
   - Print a summary of what was downloaded. Add a note in README crediting FATURA (CC BY 4.0) and Northwind.
3. Makefile: `make scanned`, `make public-data`.

Do not convert FATURA annotations into our schema (not needed now).
Finish with a summary and verification commands.
```

**Verify:** `make scanned && make public-data`. Phase 1 complete — commit and tag `phase-1`.
