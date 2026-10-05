# CLAUDE.md — po-extractor

Low-latency, high-accuracy purchase order (PO) extraction system. Interview demo built to production standards.
Full requirements: `docs/PRD.md`. Build plan: `docs/PLAN.md`. Only implement the step you are asked to implement.

## Priorities (in order)
1. **Accuracy** — never trade correctness for speed without evidence. Never pass a PO that fails validation as valid.
2. **Latency** — p95 < 30 s end to end. Minimise LLM output tokens; maximise parallelism.
3. **Clarity** — simple, readable code an interviewer can follow. No clever abstractions.

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
- The app calls **only LiteLLM** using model aliases `po-fast`, `po-accurate`, `po-baseline`. Never call vLLM or llama.cpp directly from app code.
- Every LLM call: JSON-schema constrained, `temperature=0`, thinking disabled via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` (except baseline when configured), records latency + token usage.
- Prompt order is fixed for prefix caching: system → schema rules → full PO text → group task. Do not reorder.
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

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**This project has a knowledge graph. Start with the code-review-graph
MCP tools to narrow scope, then read the source.** The graph is cheaper than scanning files and
gives you structural context (callers, dependents, test coverage) that file search cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

### Verify in the source

- Narrow scope with the graph, then read the source. Do not change code from graph output alone.
- For any non-trivial change, read the implementation and the relevant tests before concluding.
- Verify the exact source when touching behavior, database logic, migrations, retries, fallbacks,
  recovery, or compatibility code.
- When the graph and the source disagree, the source wins. The graph may be stale or may not
  model that relationship.
- An empty graph result can mean "not indexed" or "not statically visible", not "does not exist".

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
<!-- /code-review-graph MCP tools -->
