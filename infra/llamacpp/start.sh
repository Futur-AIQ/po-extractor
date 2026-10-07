#!/usr/bin/env bash
# Start the local dev model with llama.cpp's OpenAI-compatible server on port 8081.
# The app never calls this directly; LiteLLM (infra/litellm/config.dev.yaml) routes to it.
#
# Model: LOCAL_GGUF in Hugging Face "repo:quant" form, e.g. unsloth/Qwen3.5-9B-GGUF:Q4_K_M.
# Read from the environment, else from the repo-root .env. Downloaded once into the llama.cpp cache.
# -hf also downloads and loads the repo's vision projector (mmproj) when it has one; check with
#   curl -s 127.0.0.1:8081/props | jq .modalities      -> {"vision": true, ...}
#
# Slots and context come from the environment (defaults below):
#   PARALLEL   number of concurrent request slots              (default 4)
#   CTX_SIZE   KV-cache tokens shared by all slots              (default 32768)
# Baseline runs (one long multimodal request with thinking on) need the whole context for a
# single request:  PARALLEL=1 CTX_SIZE=32768 make model
set -euo pipefail

PARALLEL="${PARALLEL:-4}"
CTX_SIZE="${CTX_SIZE:-32768}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${LOCAL_GGUF:-}" && -f "$REPO_ROOT/.env" ]]; then
  # Last LOCAL_GGUF= line; strip quotes and spaces. `|| true`: a missing line is not an error here.
  LOCAL_GGUF="$( { grep -E '^LOCAL_GGUF=' "$REPO_ROOT/.env" || true; } | tail -n1 | cut -d= -f2- | tr -d "\"' ")"
fi

if [[ -z "${LOCAL_GGUF:-}" ]]; then
  echo "ERROR: LOCAL_GGUF is not set." >&2
  echo "Set it in .env (see .env.example) or the environment, in Hugging Face repo:quant form, e.g." >&2
  echo "  LOCAL_GGUF=unsloth/Qwen3.5-9B-GGUF:Q4_K_M" >&2
  exit 1
fi

if ! command -v llama-server >/dev/null 2>&1; then
  echo "ERROR: llama-server not found. Install it with: brew install llama.cpp" >&2
  exit 1
fi

echo "Starting llama-server on 127.0.0.1:8081 with $LOCAL_GGUF (parallel=$PARALLEL, ctx=$CTX_SIZE)"
# --jinja        use the model's own chat template (needed for enable_thinking + tool/JSON support)
# --parallel     concurrent slots, so fan-out requests are batched rather than queued
# --ctx-size     KV-cache tokens in total for all slots
# --kv-unified   one KV pool shared by all slots. Without it, --ctx-size is divided between
#                the slots (32768 / 4 = 8192 each), too small for a long PO prompt. With it, one
#                request may use the whole pool, but concurrent requests compete for it.
exec llama-server \
  -hf "$LOCAL_GGUF" \
  --host 127.0.0.1 \
  --port 8081 \
  --jinja \
  --parallel "$PARALLEL" \
  --ctx-size "$CTX_SIZE" \
  --kv-unified
