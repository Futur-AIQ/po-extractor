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

## Local model and LiteLLM

The app never calls a model server directly. It calls the **LiteLLM proxy** on `:4000` using
three aliases: `po-fast`, `po-accurate` and `po-baseline`. LiteLLM decides which backend serves
each alias.

```
app ──► LiteLLM :4000 ──► llama.cpp :8081  (Mac dev:   infra/litellm/config.dev.yaml)
                     └──► vLLM :8000        (H100 run:  infra/litellm/config.h100.yaml, via SSH tunnel)
```

One-time setup in `.env` (see `.env.example`):

```bash
echo "LITELLM_API_KEY=sk-$(openssl rand -hex 32)" >> .env   # proxy master key = app's bearer token
echo "LOCAL_GGUF=unsloth/Qwen3.5-9B-GGUF:Q4_K_M" >> .env    # dev model (see .env.example)
```

Then, in two terminals:

| Command | What it does |
|---|---|
| `make model` | `infra/llamacpp/start.sh`: `llama-server -hf $LOCAL_GGUF` on 127.0.0.1:8081 with `--jinja --parallel 4 --ctx-size 32768 --kv-unified`. The first run downloads the model. |
| `make litellm` | LiteLLM proxy on 127.0.0.1:4000 with `config.dev.yaml`: all three aliases go to llama.cpp, with a 120 s timeout and 2 retries |

Check it with `uv run python -m scripts.smoke_llm`. It sends one `po-fast` call
(temperature 0, `response_format` JSON schema, `chat_template_kwargs.enable_thinking=false`)
and prints the parsed JSON, latency and token counts.

**Switching to the H100 later:** only the LiteLLM config changes. Start vLLM on the H100, open
the SSH tunnel to `localhost:8000`, set `VLLM_API_KEY` in `.env`, then run `make litellm-h100`
instead of `make litellm`. App code, aliases and `.env` app settings stay the same.
`config.h100.yaml` is a template until Phase 7 (see its TODOs).

## Datasets

| Command | Output |
|---|---|
| `make dataset` | 60 synthetic Indian GST POs (`data/synthetic`): PDFs, ground truth, mock ERP masters, splits |
| `make scanned` | Scanned twin of every PO (`PO_0001_S.pdf`, image-only pages at 200 dpi) and 10 mixed POs (`PO_00xx_M.pdf`, 1-2 pages scanned) |
| `make public-data` | Public datasets below, into `data/northwind` and `data/fatura` (one-off developer download) |

`manifest.csv` has one row per PDF with a `variant` column (`N` native, `S` scanned twin,
`M` mixed) and `page_kinds`; `splits.json` lists each variant per split, so a benchmark can
sample native and scanned 50/50.

### Credits
- **FATURA Dataset** (invoice images, 50 templates): Mahmoud Limam, Marwa Dhiaf, Yousri
  Kessentini, Zenodo record [10371464](https://zenodo.org/records/10371464) (2023), licensed
  **CC BY 4.0**. We use a 50-image sample (one white-background image per template) with its
  original-format annotations; the images are unmodified.
- **Northwind purchase orders**: Hugging Face dataset
  [AyoubChLin/northwind_PurchaseOrders](https://huggingface.co/datasets/AyoubChLin/northwind_PurchaseOrders)
  by Ayoub Cherguelaine and Faycal Boubekri (Apache-2.0); English purchase-order PDFs, used
  as a smoke test.
