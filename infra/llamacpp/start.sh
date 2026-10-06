#!/usr/bin/env bash
# Start the local dev model with llama.cpp's OpenAI-compatible server on port 8081.
# The app never calls this directly; LiteLLM (infra/litellm/config.dev.yaml) routes to it.
#
# Model: LOCAL_GGUF in Hugging Face "repo:quant" form, e.g. unsloth/Qwen3.5-9B-GGUF:Q4_K_M.
# Read from the environment, else from the repo-root .env. Downloaded once into the llama.cpp cache.
set -euo pipefail

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

echo "Starting llama-server on 127.0.0.1:8081 with $LOCAL_GGUF"
# --jinja        use the model's own chat template (needed for enable_thinking + tool/JSON support)
# --parallel 4   4 concurrent slots, so fan-out requests are batched rather than queued
# --ctx-size     32768 tokens in total for all slots
# --kv-unified   one KV pool shared by all slots. Without it, setting --parallel splits the
#                context into 4 x 8192 tokens, too small for a long PO prompt.
exec llama-server \
  -hf "$LOCAL_GGUF" \
  --host 127.0.0.1 \
  --port 8081 \
  --jinja \
  --parallel 4 \
  --ctx-size 32768 \
  --kv-unified
