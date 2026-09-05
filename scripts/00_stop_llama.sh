#!/usr/bin/env bash
# Kill any running llama-server process to free VRAM.
# Idempotent: safe to run even if no llama-server is running.
set -euo pipefail

# pgrep filter excludes:
#   - 'earlyoom' (whose --avoid flag contains "llama-server")
#   - the running pgrep/grep processes themselves
if pgrep -af 'llama-server' | grep -vE '(earlyoom|grep|pgrep) ' | grep -q 'llama-server\b'; then
  echo "[INFO] | $(date -Iseconds) | stop_llama | killing llama-server process(es):"
  pgrep -af 'llama-server' | grep -vE '(earlyoom|grep|pgrep) ' | grep 'llama-server\b'
  pkill -TERM -f 'llama-server' || true
  sleep 2
  pkill -KILL -f 'llama-server' 2>/dev/null || true
  echo "[INFO] | $(date -Iseconds) | stop_llama | done"
else
  echo "[INFO] | $(date -Iseconds) | stop_llama | no llama-server process running"
fi
