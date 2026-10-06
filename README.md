# po-extractor

Low-latency, high-accuracy purchase order (PO) extraction system.
Requirements: [`docs/PRD.md`](docs/PRD.md) · Build plan: [`docs/PLAN.md`](docs/PLAN.md).

## Local services

App services run in Docker from `infra/docker-compose.yml`. For now that is a single
Redis 7 instance (container `po-redis`) used as the job queue. It listens on host port
**6380** so it does not collide with Langfuse's own Redis. Append-only persistence is on and
data lives in the `po-extractor_redis-data` volume, so queued jobs survive restarts.

| Command     | What it does                                                        |
|-------------|---------------------------------------------------------------------|
| `make up`   | Start Redis in the background and wait until its healthcheck passes |
| `make ping` | Run `redis-cli ping` inside the container; prints `PONG` when ready |
| `make logs` | Follow the service logs (Ctrl+C to stop)                            |
| `make down` | Stop and remove the container; the data volume is kept              |

To wipe Redis data as well: `docker compose -f infra/docker-compose.yml down -v`.

Tracing uses a self-hosted Langfuse (UI on **3000**), run separately with `make langfuse-up`,
`make langfuse-logs` and `make langfuse-down`. Setup, ports and the one-time UI steps are in
[`infra/langfuse/README.md`](infra/langfuse/README.md).
