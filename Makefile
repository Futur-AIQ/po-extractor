.PHONY: help install test lint format up down logs ping

COMPOSE := docker compose -f infra/docker-compose.yml

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

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
