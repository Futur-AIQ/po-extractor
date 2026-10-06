# Langfuse (self-hosted)

Tracing runs on a self-hosted Langfuse on this Mac, so PO data never leaves the machine
(PRD §8). We use the **official Langfuse Docker Compose setup**, run unmodified:

- Docs: <https://langfuse.com/self-hosting/deployment/docker-compose>
- Repo: <https://github.com/langfuse/langfuse>. The compose file is `docker-compose.yml` at the
  repo root (Langfuse v4 images: `langfuse-web`, `langfuse-worker`, Postgres, ClickHouse,
  Redis, MinIO).

## How it is wired

| Piece | What it does |
|---|---|
| `vendor/` | Shallow clone of the upstream repo, made by `make langfuse-up` on first run. Gitignored and never edited. |
| `gen-env.sh` | Prints random values for every `# CHANGEME` secret in the official compose. |
| `langfuse.env` | Output of `gen-env.sh`, created once by `make langfuse-up`. Gitignored. |

The official compose reads every secret as `${VAR:-default}`, so we pass
`--env-file infra/langfuse/langfuse.env` instead of editing the file. Upstream asks you to
change those defaults. `langfuse.env` also sets `TELEMETRY_ENABLED=false`.
The stack runs under compose project name `po-langfuse`, so its containers and volumes are
easy to tell apart from the app's.

| Command | What it does |
|---|---|
| `make langfuse-up` | Clone (if missing), generate secrets (if missing), `docker compose up -d`, wait for `/api/public/health` |
| `make langfuse-logs` | Follow the logs of all Langfuse containers |
| `make langfuse-down` | Stop the stack; data volumes are kept |

The stack needs several GB of RAM. Set Docker Desktop's memory limit to at least 8 GB.

## Ports

Host ports published by the official compose, compared with ours (CLAUDE.md):

| Host port | Service | Bound to |
|---|---|---|
| **3000** | langfuse-web (UI + API) | all interfaces |
| 3030 | langfuse-worker | 127.0.0.1 |
| 9090 / 9091 | MinIO API / console | all interfaces / 127.0.0.1 |
| 5432 | Postgres | 127.0.0.1 |
| 6379 | Langfuse's Redis | 127.0.0.1 |
| 8123 / 9000 | ClickHouse HTTP / native | 127.0.0.1 |

Our services use 6380 (app Redis), 8080 (API), 4000 (LiteLLM), 8081 (llama.cpp) and 8000
(vLLM tunnel). **None of these clash**, and the UI is on 3000 as required, so there is no
override file.

If a port is already taken on your machine (for example a Homebrew Postgres on 5432), do
not edit `vendor/`. Create `infra/langfuse/docker-compose.override.yml` and add
`-f infra/langfuse/docker-compose.override.yml` to `LANGFUSE_COMPOSE` in the Makefile. Compose
*appends* `ports` lists when it merges files, so use the `!override` tag to replace them:

```yaml
services:
  postgres:
    ports: !override
      - 127.0.0.1:5433:5432
```

If you remap 3000 or 9090, the upstream docs also require updating `NEXTAUTH_URL` and
`LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT` (put them in `langfuse.env`).

## First-time setup (manual, in the UI)

1. `make langfuse-up` and wait for "Langfuse is ready".
2. Open <http://localhost:3000> and click **Sign up**. There is no default login; the first
   user you create is local to this instance.
3. Create an **organisation** (e.g. `po-extractor`), then a **project** (e.g. `po-extractor-dev`).
4. In the project, go to **Settings → API Keys → Create new API keys**.
5. Copy the keys into the repo-root `.env` (copy it from `.env.example` if needed):
   ```
   LANGFUSE_HOST=http://localhost:3000
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```
6. `uv run python -m scripts.smoke_trace` prints a trace URL. Open it and you should see
   `smoke-trace` with child spans `pdf_text` and `llm_group`.

## Upgrade

```bash
git -C infra/langfuse/vendor pull
docker compose -p po-langfuse --project-directory infra/langfuse/vendor \
  -f infra/langfuse/vendor/docker-compose.yml --env-file infra/langfuse/langfuse.env up -d --pull always
```

## Reset (deletes all traces, users and keys)

```bash
docker compose -p po-langfuse --project-directory infra/langfuse/vendor \
  -f infra/langfuse/vendor/docker-compose.yml --env-file infra/langfuse/langfuse.env down -v
rm infra/langfuse/langfuse.env   # optional: new secrets on next make langfuse-up
```
