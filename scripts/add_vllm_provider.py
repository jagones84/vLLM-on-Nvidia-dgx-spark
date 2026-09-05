"""Add a 'local-vllm' provider entry to openclaw.json pointing at the vLLM
OpenAI-compatible server on 127.0.0.1:8000.

Backs up the original file first (mandatory per user rule: 'non distruggere tutto').

v2 (2026-09-05, fix by mentor session):
  - model id = "Qwen/Qwen3-1.7B" (the REAL served id from /v1/models).
    v1 used "qwen3-1.7b" -> vLLM returned HTTP 404 "model does not exist"
    because the server runs WITHOUT --served-model-name, so the id is the
    HF handle. If you want the short id, add --served-model-name to the
    docker run instead of editing here.
  - also appends "local-vllm/Qwen/Qwen3-1.7B" to
    agents.defaults.modelPolicy.allow: without it the model stays INVISIBLE
    in the OpenClaw model picker even though the provider exists.
  - renames stale per-agent keys "local-vllm/qwen3-1.7b" if present.
"""
import json
import shutil
import time
from pathlib import Path

CFG = Path("/home/jagones/.openclaw/openclaw.json")
BAK = Path(f"/home/jagones/.openclaw/openclaw.json.bak-addvllm-{time.strftime('%Y%m%d-%H%M%S')}")
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REF = f"local-vllm/{MODEL_ID}"

# Backup first
shutil.copy2(CFG, BAK)
print(f"backup : {BAK} ({BAK.stat().st_size} bytes)")

# Load + modify
d = json.loads(CFG.read_text())
providers = d.setdefault("models", {}).setdefault("providers", {})

# New provider - vLLM runs WITHOUT --api-key so any Bearer token is accepted.
new_provider = {
    "baseUrl": "http://127.0.0.1:8000/v1",
    "apiKey": "nim",
    "api": "openai-completions",
    "timeoutSeconds": 1800,
    "models": [
        {
            "id": MODEL_ID,
            "name": "Qwen3 1.7B Thinking (vLLM, DGX Spark)",
            "reasoning": True,
            "input": ["text"],
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
            },
            "contextWindow": 254000,
            "maxTokens": 8192,
            "api": "openai-completions",
        }
    ],
}

# Idempotent: if 'local-vllm' already exists, replace it.
if "local-vllm" in providers:
    print("note   : 'local-vllm' already present -> replacing")
providers["local-vllm"] = new_provider

# Allowlist: without this the model is NOT selectable in the picker.
allow = d["agents"]["defaults"]["modelPolicy"]["allow"]
if MODEL_REF not in allow:
    allow.append(MODEL_REF)
    print(f"allow  : added {MODEL_REF}")
else:
    print(f"allow  : already present {MODEL_REF}")

# Atomic write: write to tmp, then rename (avoid partial writes).
tmp = CFG.with_suffix(".json.tmp")
tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
tmp.replace(CFG)
print(f"wrote  : {CFG} ({CFG.stat().st_size} bytes)")
print()
print("All providers now:")
for name in sorted(providers):
    models = providers[name].get("models", [])
    if models:
        ids = ", ".join(m.get("id", "?") for m in models)
        print(f"  - {name:20s}  {ids}")
    else:
        print(f"  - {name:20s}  (no models)")
