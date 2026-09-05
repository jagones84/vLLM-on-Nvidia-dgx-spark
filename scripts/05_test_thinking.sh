#!/usr/bin/env bash
# Test the vLLM server with a thinking model and a non-thinking prompt.
# Verifies that 'reasoning_content' (think) and 'content' are separated.
# Reference: https://build.nvidia.com/spark/vllm/instructions
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

OUT_DIR="${SCRIPT_DIR}/../outputs/$(date -u +%Y%m%d_%H%M%S)_vllm_test"
mkdir -p "${OUT_DIR}"
echo "[INFO] | $(date -Iseconds) | test_vllm | outputs -> ${OUT_DIR}"

# Wait up to 900s for /health to be ready.
echo "[INFO] | $(date -Iseconds) | test_vllm | waiting for /health (max 900s)"
if ! timeout 900 bash -c 'until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 5; done'; then
  echo "[ERROR] | $(date -Iseconds) | test_vllm | server failed to start in 900s" >&2
  docker logs "${CONTAINER_NAME}" | tail -200 > "${OUT_DIR}/server_failure.log"
  exit 1
fi
echo "[INFO] | $(date -Iseconds) | test_vllm | /health is up"

# Test 1: simple non-reasoning chat (math problem to force thinking).
PROMPT_MATH='A train leaves station A at 9:00 AM traveling at 60 km/h. Another train leaves station B (300 km away) at 10:00 AM traveling toward A at 90 km/h. At what time do they meet?'

cat > "${OUT_DIR}/01_request_math.json" << JSON_EOF
{
  "model": "${MODEL_HANDLE}",
  "messages": [{"role": "user", "content": "${PROMPT_MATH}"}],
  "max_tokens": 4096,
  "temperature": 0.6,
  "top_p": 0.95,
  "chat_template_kwargs": {"enable_thinking": true}
}
JSON_EOF

echo "[INFO] | $(date -Iseconds) | test_vllm | request 1: math word problem (forces thinking)"
curl -sS http://localhost:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @"${OUT_DIR}/01_request_math.json" \
  -o "${OUT_DIR}/01_response_math.json" \
  -w "\nHTTP_STATUS:%{http_code}\n"

# Test 2: reasoning-tagged prompt to make the split explicit.
PROMPT_LOGIC='If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain step by step.'

cat > "${OUT_DIR}/02_request_logic.json" << JSON_EOF
{
  "model": "${MODEL_HANDLE}",
  "messages": [{"role": "user", "content": "${PROMPT_LOGIC}"}],
  "max_tokens": 4096,
  "temperature": 0.6,
  "top_p": 0.95,
  "chat_template_kwargs": {"enable_thinking": true}
}
JSON_EOF

echo "[INFO] | $(date -Iseconds) | test_vllm | request 2: logic reasoning"
curl -sS http://localhost:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @"${OUT_DIR}/02_request_logic.json" \
  -o "${OUT_DIR}/02_response_logic.json" \
  -w "\nHTTP_STATUS:%{http_code}\n"

# Quick validation: ensure both responses have a 'choices' array and
# the 'reasoning' (think) and 'content' are split.
# vLLM 0.28.0 puts the thinking in 'reasoning'; older vLLM used 'reasoning_content'.
python3 - <<PY_EOF > "${OUT_DIR}/03_summary.txt"
import json, os, sys
out_dir = os.environ.get("OUT_DIR", "${OUT_DIR}")
results = []
for name in ("01_response_math.json", "02_response_logic.json"):
    p = os.path.join(out_dir, name)
    with open(p) as f:
        data = json.load(f)
    if "choices" not in data or not data["choices"]:
        results.append((name, "FAIL: no choices"))
        continue
    msg = data["choices"][0].get("message", {})
    # vLLM 0.28.0 returns the think block in 'reasoning'; some clients still read
    # 'reasoning_content'. Support both for forward-compat.
    think = msg.get("reasoning") or msg.get("reasoning_content")
    content = msg.get("content")
    finish = data["choices"][0].get("finish_reason")
    usage = data.get("usage", {})
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    has_think = think is not None and len(str(think)) > 0
    has_content = content is not None and len(str(content)) > 0
    results.append((name, f"finish={finish} | has_think={has_think} ({len(str(think)) if has_think else 0} chars) | has_content={has_content} ({len(str(content)) if has_content else 0} chars) | reasoning_tokens={reasoning_tokens}"))
print("=" * 60)
print("vLLM THINKING TEST SUMMARY (Qwen3-1.7B, reasoning-parser=qwen3)")
print("=" * 60)
for name, status in results:
    print(f"{name}: {status}")
print()
print("Sample (first 400 chars of reasoning + content) from 01_response_math.json:")
with open(os.path.join(out_dir, "01_response_math.json")) as f:
    d = json.load(f)
msg = d["choices"][0]["message"]
think = msg.get("reasoning") or msg.get("reasoning_content") or ""
print("--- reasoning (THINK) ---")
print(think[:400])
print("--- content (ANSWER) ---")
print((msg.get("content") or "")[:400])
PY_EOF

cat "${OUT_DIR}/03_summary.txt"
echo
echo "[INFO] | $(date -Iseconds) | test_vllm | artifacts written to ${OUT_DIR}"
