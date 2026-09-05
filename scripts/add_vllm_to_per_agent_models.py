"""Add 'local-vllm' provider to every per-agent models.json under
~/.openclaw/agents/*/models.json. These files are the per-agent model
catalog used by the webchat picker (in addition to the global
openclaw.json providers).

Backs up each file before editing. Idempotent.
"""
import json
import shutil
import time
from pathlib import Path
import subprocess

AGENTS_DIR = Path("/home/jagones/.openclaw/agents")
TS = time.strftime("%Y%m%d-%H%M%S")

VLLM_PROVIDER = {
    "baseUrl": "http://127.0.0.1:8000/v1",
    "apiKey": "nim",
    "api": "openai-completions",
    "timeoutSeconds": 1800,
    "models": [
        {
            "id": "qwen3-1.7b",
            "name": "Qwen3 1.7B Thinking (vLLM, DGX Spark)",
            "reasoning": True,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 8192,
            "maxTokens": 8192,
            "api": "openai-completions",
        }
    ],
}

targets = sorted(AGENTS_DIR.rglob("models.json"))
print(f"found {len(targets)} per-agent models.json files")
print()

for mf in targets:
    bak = mf.with_suffix(f".json.bak-vllm-{TS}")
    shutil.copy2(mf, bak)
    d = json.loads(mf.read_text())
    providers = d.setdefault("providers", {})
    if "local-vllm" in providers:
        providers["local-vllm"] = VLLM_PROVIDER
        action = "replaced"
    else:
        providers["local-vllm"] = VLLM_PROVIDER
        action = "added"
    # atomic write
    tmp = mf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(mf)
    print(f"  [{action:8s}] {mf}  (backup: {bak.name})")

print()
print("Verifying gateway hot-reload...")
import time as _t
_t.sleep(2)
out = subprocess.run(
    ["journalctl", "--user", "-u", "openclaw-gateway", "--since", "20s ago", "--no-pager"],
    capture_output=True, text=True
).stdout
hits = [ln for ln in out.splitlines() if "reload" in ln.lower() and "local-vllm" in ln.lower()]
if hits:
    for ln in hits[-5:]:
        print(f"  {ln.rstrip()}")
else:
    print("  (no reload line yet — gateway may auto-detect the change)")
