"""Add a 'local-vllm' provider entry to openclaw.json pointing at the vLLM
OpenAI-compatible server on 127.0.0.1:8000.

Backs up the original file first (mandatory per user rule: 'non distruggere tutto').
"""
import json
import shutil
import time
from pathlib import Path

CFG = Path("/home/jagones/.openclaw/openclaw.json")
BAK = Path(f"/home/jagones/.openclaw/openclaw.json.bak-addvllm-{time.strftime('%Y%m%d-%H%M%S')}")

# Backup first
shutil.copy2(CFG, BAK)
print(f"backup : {BAK} ({BAK.stat().st_size} bytes)")

# Load + modify
d = json.loads(CFG.read_text())
providers = d.setdefault("models", {}).setdefault("providers", {})

# New provider — follows the same schema as local-flashnext-unc.
# Note: vLLM is currently running WITHOUT --api-key so any Bearer token is
# accepted; the 'nim' literal here matches VLLM_API_KEY in gateway.systemd.env
# (the canonical NVIDIA NIM default) so the same key can be used by other
# clients too.
new_provider = {
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
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
            },
            "contextWindow": 8192,
            "maxTokens": 8192,
            "api": "openai-completions",
        }
    ],
}

# Idempotent: if 'local-vllm' already exists, replace it.
if "local-vllm" in providers:
    print("note   : 'local-vllm' already present -> replacing")
providers["local-vllm"] = new_provider

# Atomic write: write to tmp, then rename (avoid partial writes).
tmp = CFG.with_suffix(".json.tmp")
tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
tmp.replace(CFG)
print(f"wrote  : {CFG} ({CFG.stat().st_size} bytes)")
print(f"new    : providers['local-vllm'].models[0].id = {providers['local-vllm']['models'][0]['id']!r}")
print()
print("All providers now:")
for name in sorted(providers):
    models = providers[name].get("models", [])
    if models:
        ids = ", ".join(m.get("id", "?") for m in models)
        print(f"  - {name:20s}  {ids}")
    else:
        print(f"  - {name:20s}  (no models)")
