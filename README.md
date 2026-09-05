# vLLM on DGX Spark

Containerized vLLM serving of small thinking models on NVIDIA DGX Spark (GB10, SM\_121, 128GB unified memory).

Follows the official recipe: <https://build.nvidia.com/spark/vllm/instructions>
and the model recipe: <https://recipes.vllm.ai/Qwen/Qwen3-1.7B>

## Layout

```
vLLM/                        (this repo, the scripts + docs only)
├── .agent/                  # documentation (HANDOFF, domain manuals)
├── scripts/                 # operational scripts (numbered 00-07)
├── outputs/                 # SYMLINK -> /home/jagones/runs
├── .venv-abliteration       # SYMLINK -> /home/jagones/.venvs/vllm-abliteration
├── trash/                   # scratch / one-off experiments (excluded from mass fixes)
│   └── abliteration_workspace/  # clones of upstream recipes (gitignored)
└── logs/                    # long-running log captures

Heavy artifacts live OUTSIDE the repo (per the 2026-09-05 housekeeping):
  /home/jagones/models/      # downloaded model directories (always)
  /home/jagones/runs/        # per-run test logs (target of outputs/ symlink)
  /home/jagones/.venvs/      # Python virtualenvs (target of .venv-* symlinks)
```

## Quickstart

### Default: Qwen3-1.7B @ 254K (thinking, YaRN)

```bash
# 0. (only if a llama-server is hogging VRAM)
bash /home/jagones/Programs/vLLM/scripts/00_stop_llama.sh

# 1. Pull the vLLM image (one-shot)
bash /home/jagones/Programs/vLLM/scripts/01_pull_vllm.sh

# 2. Start the server detached
bash /home/jagones/Programs/vLLM/scripts/02b_start_vllm_detached.sh

# 3. Tail logs (first start takes ~2 min including torch.compile + cudagraph capture)
bash /home/jagones/Programs/vLLM/scripts/04_logs.sh

# 4. Test with a reasoning prompt that triggers 'think' + 'content' split
bash /home/jagones/Programs/vLLM/scripts/05_test_thinking.sh

# 5. Stop the server
bash /home/jagones/Programs/vLLM/scripts/03_stop_vllm.sh
```

### Alternative: DeepSeek-V4-Flash-0731 (abliterated, 80 GB, 131K ctx)

```bash
# Stop the Qwen3 server first (one container at a time on :8000)
bash /home/jagones/Programs/vLLM/scripts/03_stop_vllm.sh

# Start the abliterated EXL3 K2 model in the tpurtell container
bash /home/jagones/Programs/vLLM/scripts/02c_start_tpurtell_abliterated.sh

# Bench (decode + reasoning separation + prefill test)
ssh dgx "python3 /home/jagones/Programs/vLLM/trash/bench_tps.py --long 100000"

# Smoke-test uncensored compliance
ssh dgx "python3 /home/jagones/Programs/vLLM/trash/test_uncensored.py"

# Stop
bash /home/jagones/Programs/vLLM/scripts/02c_start_tpurtell_abliterated.sh --stop
```

Server profile lives in `trash/abliteration_workspace/tpurtell/.env`:
131,072-token context, fixed **13.5 GiB KV pool** via
`--kv-cache-memory-bytes` (EXTRA_VLLM_ARGS), `MODE=off` (no dSpark
draft), gpu-mem 0.80. Measured: **19.3 tok/s decode, 1044.9 tok/s
prefill @ 111K, ~95 GiB / 121.69 GiB device**.

**DO NOT** use the recipe's aarch64 defaults (1M ctx, 0.85+ util) —
they OOM-kill the host via `earlyoom` (restart loop). On a DGX Spark
that also runs a desktop + gateway, keep util at 0.80 and pin the KV
pool instead of raising the fraction. See `.agent/HANDOFF.md`
lessons #14-19.

**Think / content separation:** the model serves with
`--reasoning-parser deepseek_v4`; pass
`"chat_template_kwargs": {"thinking": true}` per request to receive
`message.reasoning` (trace) and `message.content` (answer) as
separate fields.

## Models on the DGX Spark

| Path                                            | Size  | Role                                                                                  |
| ----------------------------------------------- | ----- | ------------------------------------------------------------------------------------- |
| `/home/jagones/models/exl3-k2-abliterated/`     | 79 GB | **CURRENTLY RUNNING**. EXL3 K2 quantized model with 92 `wo_b` tensors swapped to abliterated values. 8/8 AdvBench probes passed. |
| `/home/jagones/models/exl3-k2-stock/`           | 79 GB | Pristine EXL3 K2 quantized model (untouched, for rollback or comparison).              |

~~`/home/jagones/models/dsv4-ablit-safetensors/` (157 GB)~~ deleted on
2026-09-05 after end-to-end verification. It was the BF16 source
overlay (92 `wo_b` tensors) from
`cebeuq/DeepSeek-V4-Flash-0731-abliterated`; re-download it from
HuggingFace and re-run `trash/swap_wob.py` if the abliterated output
ever needs rebuilding.

The two remaining directories are NOT two different models — they are
**censored stock vs its abliterated twin** (identical except the 92
swapped `wo_b` tensors).

## Default model (when using Qwen3 recipe)

`Qwen/Qwen3-1.7B` - smallest Qwen3 family member with native thinking,
running at **254K context** (40K native + YaRN factor 6.4). Swap in
`Qwen/Qwen3-4B` (8GB fp16) or `Qwen/Qwen3-8B` (16GB fp16) for more
capability by editing `scripts/env.sh` (`MODEL_HANDLE`). See
`.agent/HANDOFF.md` § "Current state" for live config.

## Tuning knobs (env.sh)

| Variable        | Default                   | Notes                                      |
| --------------- | ------------------------- | ------------------------------------------ |
| `MODEL_HANDLE`  | `Qwen/Qwen3-1.7B`         | Any HF model vLLM supports                 |
| `VLLM_IMAGE`    | `vllm/vllm-openai:latest` | NVIDIA recipe-recommended tag              |
| `MAX_MODEL_LEN` | `254000`                  | prompt+output; YaRN-extended from native 40K |
| `GPU_MEM_UTIL`  | `0.28`                    | fraction of GPU vLLM may consume (lean for 254K) |
| `ROPE_SCALING`  | `{"rope_type":"yarn","factor":6.4,...}` | YaRN config, injected via `--hf-overrides` |
| `VLLM_PORT`     | `8000`                    | host port (avoid 8130 = llama-server)      |

## Why `--reasoning-parser qwen3`?

vLLM splits the model's hidden thinking trace from the user-visible answer
when a reasoning parser is enabled. The chat completion returns two
fields per message in vLLM 0.28.0:

- `message.reasoning` - the model's internal "think" block (NEW in 0.28.0; older
  releases exposed it as `reasoning_content`).

- `message.content` - the user-facing answer.

Token usage also reports the split via
`usage.completion_tokens_details.reasoning_tokens` (how many tokens went
into the think block).

Without a parser, the think tokens end up concatenated inside `content` and
the client cannot separate them.

## Use the model from openclaw

The vLLM server is OpenAI-compatible on `http://127.0.0.1:8000/v1`. To make
it available as a selectable model inside openclaw's webchat / agents,
register it as a provider:

```bash
bash scripts/06_register_openclaw.sh
```

This adds a `local-vllm` entry under `models.providers` in
`~/.openclaw/openclaw.json`, pointing to the vLLM endpoint. The gateway
hot-reloads the file (no service restart) and the model appears as
**"Qwen3 1.7B Thinking (vLLM, DGX Spark)"** in the model picker.

Roll back with the timestamped backup the script creates
(`openclaw.json.bak-addvllm-<TS>`).

## vRAM check

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

## License

Apache 2.0 — see [LICENSE](LICENSE). The vLLM server itself is also
Apache 2.0 (NVIDIA/UCLA), so this repo inherits the same license
family for consistency.

## Known issues (DGX Spark, sm\_121)

See `.agent/HANDOFF.md` for the current state (DeepSeek-V4-Flash-0731
abliterated running, Qwen3-1.7B @ 254K available) and
`.agent/README-vllm-dgx.md` for the deep manual, including the
abliteration pipeline and the diagnosis cheatsheet.

The most common breakage modes and their fixes are summarized in
the cheatsheet at the bottom of `.agent/README-vllm-dgx.md`:
`--enforce-eager`, `--rope-scaling` vs `--hf-overrides`, OpenClaw
picker empty / 404, **OOM-kill of the tpurtell container via
`earlyoom`** (lesson #14), etc.
