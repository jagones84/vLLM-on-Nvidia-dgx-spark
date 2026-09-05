#!/usr/bin/env bash
# Stop and remove the vLLM container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  echo "[INFO] | $(date -Iseconds) | stop_vllm | stopping ${CONTAINER_NAME}"
  docker stop "${CONTAINER_NAME}"
  docker rm "${CONTAINER_NAME}"
  echo "[INFO] | $(date -Iseconds) | stop_vllm | done"
else
  echo "[INFO] | $(date -Iseconds) | stop_vllm | no container named ${CONTAINER_NAME}"
fi
