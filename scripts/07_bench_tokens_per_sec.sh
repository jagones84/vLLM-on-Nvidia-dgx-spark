#!/usr/bin/env bash
# Benchmark vLLM throughput: TTFT, generation tok/s, end-to-end tok/s.
# Outputs to outputs/<date>_bench/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

OUT_DIR="${SCRIPT_DIR}/../outputs/$(date -u +%Y%m%d_%H%M%S)_bench"
mkdir -p "${OUT_DIR}"
echo "[INFO] | $(date -Iseconds) | bench | out -> ${OUT_DIR}"

echo "[INFO] | $(date -Iseconds) | bench | waiting for /health (max 900s)"
if ! timeout 900 bash -c 'until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 5; done'; then
  echo "[ERROR] | $(date -Iseconds) | bench | server failed to start in 900s" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/_bench_tokens_per_sec.py" \
  --out-dir "${OUT_DIR}" \
  --port "${VLLM_PORT}" \
  --model "${MODEL_HANDLE}" \
  --repeats 3

echo
echo "[INFO] | $(date -Iseconds) | bench | done. artifacts in ${OUT_DIR}"
