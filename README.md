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

## Default model

`Qwen/Qwen3-1.7B` - smallest Qwen3 family member with native thinking,
running at **254K context** (40K native + YaRN factor 6.4). Swap in
`Qwen/Qwen3-4B` (8GB fp16) or `Qwen/Qwen3-8B` (16GB fp16) for more
capability by editing `scripts/env.sh` (`MODEL_HANDLE`). See
`.agent/HANDOFF.md` § "Current state" for live config.

**Abliterated variants** (DeepSeek-V4-Flash-0731 Uncensored, EXL3 K2)
are built via the tensor-swap pipeline described in
`.agent/README-vllm-dgx.md` § "Abliteration pipeline". Output lives
in `/home/jagones/models/exl3-k2-abliterated/`, requires the
tpurtell GHCR container (sm_121 SparkInfer/ExLlamaV3/Trellis
kernels).

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

See `.agent/HANDOFF.md` for the current state (Qwen3-1.7B @ 254K live)
and `.agent/README-vllm-dgx.md` for the deep manual, including the
abliteration pipeline and the diagnosis cheatsheet.

The most common breakage modes and their fixes are summarized in
the cheatsheet at the bottom of `.agent/README-vllm-dgx.md`:
`--enforce-eager`, `--rope-scaling` vs `--hf-overrides`, OpenClaw
picker empty / 404, etc.
