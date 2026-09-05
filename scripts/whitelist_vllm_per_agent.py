"""Add 'local-vllm/qwen3-1.7b' to the models whitelist of every agent in
openclaw.json so it appears in the webchat model picker for all agents,
not just coordinator. Idempotent.

Backs up the file first.
"""
import json
import shutil
import time
from pathlib import Path

CFG = Path("/home/jagones/.openclaw/openclaw.json")
TS = time.strftime("%Y%m%d-%H%M%S")
BAK = Path(f"{CFG}.bak-vllmwhitelist-{TS}")
shutil.copy2(CFG, BAK)
print(f"backup : {BAK} ({BAK.stat().st_size} bytes)")

d = json.loads(CFG.read_text())
entries = d["agents"]["entries"]
NEW_KEY = "local-vllm/qwen3-1.7b"
changed = []

for name, ag in entries.items():
    models_field = ag.get("models")
    if models_field is None:
        # Add a fresh whitelist
        ag["models"] = {NEW_KEY: {}}
        changed.append(f"{name}: created whitelist with {NEW_KEY}")
    elif isinstance(models_field, dict):
        if NEW_KEY not in models_field:
            models_field[NEW_KEY] = {}
            changed.append(f"{name}: appended {NEW_KEY} (now {len(models_field)} whitelisted)")
        else:
            changed.append(f"{name}: {NEW_KEY} already present (no-op)")

# Atomic write
tmp = CFG.with_suffix(".json.tmp")
tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
tmp.replace(CFG)
print(f"wrote  : {CFG} ({CFG.stat().st_size} bytes)")
print()
for line in changed:
    print(f"  {line}")

print()
print("Gateway hot-reload status (last 10s of journal):")
import subprocess
out = subprocess.run(
    ["journalctl", "--user", "-u", "openclaw-gateway", "--since", "10s ago", "--no-pager"],
    capture_output=True, text=True
).stdout
for line in out.splitlines():
    if any(s in line for s in ("local-vllm", "qwen3-1.7b", "reload", "whitelist", "models")):
        print(f"  {line.rstrip()}")
