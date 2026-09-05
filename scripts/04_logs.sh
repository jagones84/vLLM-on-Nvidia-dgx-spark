#!/usr/bin/env bash
# Tail logs from the vLLM container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

docker logs -f "${CONTAINER_NAME}"
