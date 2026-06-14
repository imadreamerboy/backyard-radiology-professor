#!/usr/bin/env bash
set -euo pipefail

uv run --no-sync python scripts/prepare_runtime.py

llama-server \
  --models-preset "${RAD_TRAINER_LLAMA_PRESET:-runtime/models.ini}" \
  --host 127.0.0.1 \
  --port 8080 \
  --metrics \
  --models-max 1 \
  --api-key "${RAD_TRAINER_LLAMA_API_KEY:-local}" \
  > outputs/llama-router.log 2>&1 &
LLAMA_PID=$!

cleanup() {
  kill "$LLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

uv run --no-sync python scripts/smoke_llama_router.py \
  --wait-seconds 900 \
  --timeout-seconds 900
exec uv run --no-sync python app.py
