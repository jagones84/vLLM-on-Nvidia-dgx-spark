#!/usr/bin/env bash
# Start vLLM in fully-detached mode and exit (no logs tail).
# Uses the OFFICIAL recipe from https://recipes.vllm.ai/Qwen/Qwen3-1.7B
# for DGX Spark (GB10): minimal flags, no enforce-eager.
set -euo pipefail

SCRIPT_DIR="/home/jagones/Programs/vLLM/scripts"
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${HF_CACHE_HOST}/hub"

# Detect a real llama-server process (excludes earlyoom's "--avoid llama-server" arg,
# any grep/pgrep doing the check, and any other ps lines that merely mention the string).
if pgrep -af 'llama-server' | grep -vE '(earlyoom|grep|pgrep) ' | grep -q 'llama-server\b'; then
  echo "[ERROR] | $(date -Iseconds) | start_vllm_bg | llama-server is running. Stop it first." >&2
  pgrep -af 'llama-server' | grep -vE '(earlyoom|grep|pgrep) ' || true
  exit 1
fi

# If a vllm-server container is still around, remove it.
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "[INFO] | $(date -Iseconds) | start_vllm_bg | launching ${MODEL_HANDLE}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "" \
  -p "${VLLM_PORT}:8000" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1200 \
  -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
  "${VLLM_IMAGE}" \
  vllm serve "${MODEL_HANDLE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 8000

echo "[INFO] | $(date -Iseconds) | start_vllm_bg | container started, id:"
docker ps --filter "name=${CONTAINER_NAME}" --format "{{.ID}} {{.Status}}"
