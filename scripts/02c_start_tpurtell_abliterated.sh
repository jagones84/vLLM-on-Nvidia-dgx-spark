#!/usr/bin/env bash
# 02c_start_tpurtell_abliterated.sh
# -------------------------------------------------------------------------
# Start the tpurtell DeepSeek-V4-Flash-0731 EXL3 K2 (abliterated) recipe
# on the DGX Spark.
#
# The single source of truth for the server profile is the compose `.env`
# file next to compose.yaml:
#   trash/abliteration_workspace/tpurtell/.env
#
# Current profile (2026-09-05, proven healthy + benchmarked):
#   MODE=off                          # dSpark draft disabled (-5.5 GiB)
#   MAX_MODEL_LEN=131072              # 131K context
#   GPU_MEMORY_UTILIZATION=0.80       # proven safe against earlyoom
#   EXTRA_VLLM_ARGS=--kv-cache-memory-bytes 14500000000
#                                     # fixed 13.5 GiB KV pool (530K tokens)
#   PREFIX_CACHE=0, MAX_NUM_SEQS=2, MAX_NUM_BATCHED_TOKENS=2048
#
# Measured on GB10 (128 GB unified):
#   decode 19.3 tok/s | prefill 1044.9 tok/s @ 111K | ~95 GiB GPU total
#
# WARNING: do NOT raise GPU_MEMORY_UTILIZATION to the recipe default
# (0.85+) or remove the kv cap: earlyoom kills EngineCore when host
# free memory drops below 5% (see .agent/HANDOFF.md lessons #14-15).
# -------------------------------------------------------------------------
set -Eeuo pipefail

DGX_HOST=${DGX_HOST:-dgx}
TPURTELL_DIR=${TPURTELL_DIR:-/home/jagones/Programs/vLLM/trash/abliteration_workspace/tpurtell}
MODEL_PATH=${MODEL_PATH:-/home/jagones/models/exl3-k2-abliterated}
CONTAINER=tpurtell-deepseek-v4-flash-1

if [[ "${1:-}" == "--stop" ]]; then
  ssh "${DGX_HOST}" "docker rm -f ${CONTAINER} 2>/dev/null || true"
  echo ">>> stopped"
  exit 0
fi

# Sanity: the abliterated model dir must exist on the DGX.
if ! ssh "${DGX_HOST}" "test -d ${MODEL_PATH}"; then
  echo "ERROR: abliterated model dir not found at ${MODEL_PATH}" >&2
  exit 2
fi

# Sanity: the recipe dir with compose.yaml + .env must exist.
if ! ssh "${DGX_HOST}" "test -f ${TPURTELL_DIR}/compose.yaml && test -f ${TPURTELL_DIR}/.env"; then
  echo "ERROR: ${TPURTELL_DIR} must contain compose.yaml AND .env" >&2
  exit 2
fi

# Idempotent restart.
ssh "${DGX_HOST}" "docker rm -f ${CONTAINER} 2>/dev/null || true"

echo ">>> launching tpurtell abliterated recipe on ${DGX_HOST}"
ssh "${DGX_HOST}" "cd ${TPURTELL_DIR} && docker compose up -d 2>&1 | tail -3"

echo
echo ">>> waiting for APIServer healthcheck (up to 7.5 min) ..."
ok=""
for i in $(seq 1 90); do
  status=$(ssh "${DGX_HOST}" "docker inspect ${CONTAINER} --format '{{.State.Health.Status}}' 2>/dev/null" || true)
  if [[ "${status}" == "healthy" ]]; then
    echo ">>> healthy after ~$((i * 5))s"
    ok=1
    break
  fi
  if (( i == 90 )); then
    break
  fi
  sleep 5
done

if [[ -z "${ok}" ]]; then
  echo "ERROR: container did not become healthy." >&2
  echo "  Check: ssh ${DGX_HOST} docker logs --tail 100 ${CONTAINER}" >&2
  echo "  OOM?   ssh ${DGX_HOST} 'journalctl --since \"15 minutes ago\" | grep -i oom'" >&2
  exit 1
fi

echo
echo ">>> model: deepseek-v4-flash-0731-exl3-k2  on http://${DGX_HOST}:8000"
echo ">>> logs:      ssh ${DGX_HOST} docker logs -f ${CONTAINER}"
echo ">>> bench:     ssh ${DGX_HOST} python3 /home/jagones/Programs/vLLM/trash/bench_tps.py"
echo ">>> uncensor:  ssh ${DGX_HOST} python3 /home/jagones/Programs/vLLM/trash/test_uncensored.py"
echo ">>> stop:      bash $0 --stop"
