#!/usr/bin/env bash
# Pull the vLLM Docker image as defined in env.sh.
# Reference: https://build.nvidia.com/spark/vllm/instructions
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "[INFO] | $(date -Iseconds) | pull_vllm | pulling image: ${VLLM_IMAGE}"
docker pull "${VLLM_IMAGE}"
echo "[INFO] | $(date -Iseconds) | pull_vllm | pull complete"
docker images | grep vllm || true
