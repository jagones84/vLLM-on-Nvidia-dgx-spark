#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server in a Docker container (foreground).
# This wrapper just calls 02b (which runs detached) and then tails the logs.
# Use 02b alone for non-interactive launches.
# Reference: https://build.nvidia.com/spark/vllm/instructions
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

bash "${SCRIPT_DIR}/02b_start_vllm_detached.sh"
echo "Tailing logs (Ctrl-C to detach; the server keeps running):"
docker logs -f "${CONTAINER_NAME}"
