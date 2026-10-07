.PHONY: help install test lint format up down logs ping langfuse-up langfuse-down langfuse-logs model litellm litellm-h100 doctor dataset scanned public-data baseline eval compare

COMPOSE := docker compose -f infra/docker-compose.yml

# Langfuse: official compose from the upstream repo, run unmodified (see infra/langfuse/README.md).
LANGFUSE_REPO := https://github.com/langfuse/langfuse.git
LANGFUSE_DIR := infra/langfuse/vendor
LANGFUSE_ENV := infra/langfuse/langfuse.env
LANGFUSE_COMPOSE := docker compose -p po-langfuse --project-directory $(LANGFUSE_DIR) \
	-f $(LANGFUSE_DIR)/docker-compose.yml --env-file $(LANGFUSE_ENV)

help:  ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install runtime + dev dependencies into .venv
	uv sync

test:  ## Run the test suite
	uv run pytest

lint:  ## Check lint and formatting (no changes)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

up:  ## Start app services (Redis on :6380) and wait until healthy
	$(COMPOSE) up -d --wait

down:  ## Stop app services (data volume is kept)
	$(COMPOSE) down

logs:  ## Follow app service logs
	$(COMPOSE) logs -f

ping:  ## Check app Redis responds (expects PONG)
	docker exec po-redis redis-cli ping

$(LANGFUSE_DIR):
	git clone --depth 1 $(LANGFUSE_REPO) $(LANGFUSE_DIR)

$(LANGFUSE_ENV):
	infra/langfuse/gen-env.sh > $(LANGFUSE_ENV)
	@echo "Generated $(LANGFUSE_ENV) with random secrets"

langfuse-up: | $(LANGFUSE_DIR) $(LANGFUSE_ENV)  ## Start self-hosted Langfuse (UI on :3000)
	$(LANGFUSE_COMPOSE) up -d
	@echo "Waiting for Langfuse at http://localhost:3000 (first start can take 2-3 min)..."
	@for i in $$(seq 1 90); do \
		curl -fsS -o /dev/null http://localhost:3000/api/public/health 2>/dev/null && \
			echo "Langfuse is ready: http://localhost:3000" && exit 0; \
		sleep 2; \
	done; echo "Langfuse not ready after 3 min; check: make langfuse-logs"; exit 1

langfuse-down:  ## Stop Langfuse (data volumes are kept)
	$(LANGFUSE_COMPOSE) down

langfuse-logs:  ## Follow Langfuse logs
	$(LANGFUSE_COMPOSE) logs -f

model:  ## Start local llama.cpp on :8081 (LOCAL_GGUF from .env; env PARALLEL=4 CTX_SIZE=32768)
	infra/llamacpp/start.sh

litellm:  ## Start LiteLLM proxy on :4000 with the dev config (routes to llama.cpp)
	uv run litellm --config infra/litellm/config.dev.yaml --host 127.0.0.1 --port 4000

litellm-h100:  ## Start LiteLLM proxy on :4000 with the H100 config (vLLM via SSH tunnel)
	uv run litellm --config infra/litellm/config.h100.yaml --host 127.0.0.1 --port 4000

doctor:  ## Health check: .env, Docker, Redis, Langfuse, llama.cpp, LiteLLM + aliases
	uv run python -m scripts.doctor

dataset:  ## Build the synthetic dataset (60 POs, seed 42) into data/synthetic
	uv run playwright install chromium
	uv run python -m datagen.build --n 60 --seed 42 --out data/synthetic

scanned:  ## Scanned twins for every PO + 10 mixed POs (needs make dataset first)
	uv run python -m datagen.scanify --src data/synthetic

public-data:  ## Download Northwind (Hugging Face) and FATURA (Zenodo, ~690 MB) into data/
	uv run python -m scripts.download_public

# Baseline B0 (needs `PARALLEL=1 CTX_SIZE=32768 make model` and `make litellm`).
SPLIT ?= dev
VARIANTS ?= N,S
LIMIT ?=
CONCURRENCY ?= 1
RUN_ID ?=
baseline:  ## Baseline B0 run, e.g. make baseline SPLIT=dev VARIANTS=N,S LIMIT=3 [RUN_ID=b0_dev_local]
	uv run python -m baseline.run --split $(SPLIT) --variants $(VARIANTS) \
		--concurrency $(CONCURRENCY) $(if $(LIMIT),--limit $(LIMIT)) $(if $(RUN_ID),--run-id $(RUN_ID))

eval:  ## Score a run and write report.md/json + errors.csv, e.g. make eval RUN=runs/oracle_dev
	@test -n "$(RUN)" || { echo "usage: make eval RUN=runs/<run_id>"; exit 2; }
	uv run python -m eval.report --run $(RUN)

compare:  ## Compare runs + PRD 11.3 accept rule, e.g. make compare REF=runs/<A> CAND=runs/<B>
	@test -n "$(REF)" -a -n "$(CAND)" || { echo "usage: make compare REF=runs/<A> CAND=runs/<B>"; exit 2; }
	uv run python -m eval.compare --ref $(REF) --cand $(CAND)
