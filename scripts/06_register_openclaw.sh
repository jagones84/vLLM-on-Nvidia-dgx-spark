#!/usr/bin/env bash
# Register the running vLLM server as a model provider in openclaw.
# Idempotent: safe to re-run; replaces the existing 'local-vllm' entry if present.
#
# Usage:  bash scripts/06_register_openclaw.sh
#
# What it does:
#   1. Backs up the current openclaw.json (timestamped).
#   2. Adds a 'local-vllm' provider entry under models.providers, pointing
#      to the vLLM OpenAI-compatible server at 127.0.0.1:${VLLM_PORT:-8000}/v1.
#   3. Writes the updated JSON atomically (write-then-rename).
#   4. The openclaw gateway detects the file change and hot-reloads it
#      (no service restart required).
#
# Verify after running:
#   journalctl --user -u openclaw-gateway --since '30s ago' | grep 'local-vllm'
#   # expected line: "[reload] config hot reload applied (models.providers.local-vllm)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

OPENCLAW_JSON="${HOME}/.openclaw/openclaw.json"
TS="$(date +%Y%m%d-%H%M%S)"
BACKUP="${OPENCLAW_JSON}.bak-addvllm-${TS}"

if [[ ! -f "${OPENCLAW_JSON}" ]]; then
    echo "ERROR: ${OPENCLAW_JSON} not found. Is openclaw installed?" >&2
    exit 1
fi

echo "1) Backing up ${OPENCLAW_JSON} -> ${BACKUP}"
cp -a "${OPENCLAW_JSON}" "${BACKUP}"

# Run the python registration script in scripts/ (kept here so it ships with
# the repo; the JSON edit is non-trivial enough that bash+sed would be brittle).
REGISTER_SCRIPT="${SCRIPT_DIR}/add_vllm_provider.py"
if [[ ! -f "${REGISTER_SCRIPT}" ]]; then
    echo "ERROR: ${REGISTER_SCRIPT} not found" >&2
    exit 1
fi

echo "2) Registering local-vllm provider at http://127.0.0.1:${VLLM_PORT}/v1"
python3 "${REGISTER_SCRIPT}"

echo
echo "3) Verifying hot-reload (last 5s of gateway journal)..."
sleep 1
journalctl --user -u openclaw-gateway --since '5s ago' --no-pager 2>/dev/null \
    | grep -E 'local-vllm|reload' | tail -3 || echo "(no journal entries yet — may need a moment)"

echo
echo "Done. To use the model from openclaw webchat: open the model picker and"
echo "select 'Qwen3 1.7B Thinking (vLLM, DGX Spark)'."
