# HANDOFF — vLLM on DGX Spark

Last updated: 2026-09-05 10:18

## Current state

**Status: WORKING with 254K context via YaRN.** vLLM 0.28.0
(`vllm/vllm-openai:latest`) serves `Qwen/Qwen3-1.7B` on `localhost:8000`
on the DGX Spark, extended to 254K context (40K native -> 254K via YaRN
factor 6.4, passed via `--hf-overrides`). The reasoning parser splits
`message.reasoning` (think) from `message.content` (answer). Confirmed
end-to-end with 3 test prompts (simple, Italian, 40K long-context) on
2026-09-05. **GPU memory: 32.6 GiB (vs 97 GiB before, 3x more context
in 1/3 the memory).**

## Server

- **Image:** `vllm/vllm-openai:latest` (v0.28.0, 20.6 GB on disk)
- **Model:** `Qwen/Qwen3-1.7B` (3.78 GiB checkpoint, BF16)
- **Context:** 254000 tokens (YaRN factor 6.4 over native 40K)
- **Port:** 8000 (host) -> 8000 (container)
- **Container name:** `vllm-server`
- **GPU memory utilization:** 0.28 (lean profile for 254K)
- **Max num batched tokens:** 32768
- **First-start time:** ~140s end-to-end (autotune + YaRN warmup
  slightly longer than baseline 130s). Subsequent starts are faster
  due to on-disk compile cache.

## YaRN context extension

Qwen3-1.7B's native `max_position_embeddings` is 40960 (40K). We
extend to 254K via YaRN rope scaling. vLLM 0.28.0 does **NOT** accept
`--rope-scaling` as a CLI flag — it must be passed via `--hf-overrides`:

```bash
--hf-overrides '{"rope_scaling": {"rope_type":"yarn","factor":6.4,"original_max_position_embeddings":40960}}'
```

The variable `ROPE_SCALING` in `scripts/env.sh` holds the inner JSON
(`{"rope_type":"yarn","factor":6.4,"original_max_position_embeddings":40960}`)
and `scripts/02b_start_vllm_detached.sh` wraps it in `--hf-overrides`.

To switch profiles, change `MAX_MODEL_LEN` and `ROPE_SCALING` together:

| Target context | factor | MAX_MODEL_LEN | GPU_MEM_UTIL | total GPU |
|---|---|---|---|---|
| 65K   | 1.6 | 65000   | 0.12 | ~14 GB |
| 131K  | 3.2 | 131000  | 0.18 | ~21 GB |
| **254K** | **6.4** | **254000** | **0.28** | **~33 GB** |
| 320K  | 8.0 | 320000  | 0.35 | ~42 GB |

## Reasoning-parser output (vLLM 0.28.0 schema)

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "<final user-facing answer>",
      "reasoning": "<model's internal think block>",
      "refusal": null,
      "annotations": null
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "completion_tokens_details": {
      "reasoning_tokens": 3012
    }
  }
}
```

> Note: vLLM 0.28.0 renamed `reasoning_content` -> `reasoning`. Test
> scripts read both keys for forward-compat.

## Launch command (canonical)

See `scripts/02b_start_vllm_detached.sh`. The vLLM flags we use:

```
vllm serve Qwen/Qwen3-1.7B \
  --max-model-len 254000 \
  --gpu-memory-utilization 0.28 \
  --max-num-batched-tokens 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --hf-overrides '{"rope_scaling": {"rope_type":"yarn","factor":6.4,"original_max_position_embeddings":40960}}' \
  --host 0.0.0.0 \
  --port 8000
```

No `--enforce-eager` (let torch.compile + cudagraph capture run; the GB10
handles it fine and you need the speed).

## Quick ops

```bash
# health
curl -sf http://localhost:8000/health
# logs
bash /home/jagones/Programs/vLLM/scripts/04_logs.sh
# stop
bash /home/jagones/Programs/vLLM/scripts/03_stop_vllm.sh
# restart
bash /home/jagones/Programs/vLLM/scripts/02b_start_vllm_detached.sh
# run thinking test
bash /home/jagones/Programs/vLLM/scripts/05_test_thinking.sh
```

## Lessons learned (this session)

1. **`--enforce-eager` breaks Qwen3-1.7B on sm_121.** Symptom:
   `RuntimeError: Engine core initialization failed. ... Failed core
   proc(s): {}` ~1s after the EngineCore reports
   `init engine took 45.88 s`. Root cause: the wait function's poller
   registered a non-handshake fd that fired spuriously under eager
   mode. **Fix: remove `--enforce-eager` and let torch.compile run.**

2. **`--reasoning-parser` name matters.** Use `qwen3` for the Qwen3
   family, not `deepseek_r1`. Qwen3-1.7B's chat template emits
    `<think>...</think>` blocks; the qwen3 parser handles that.

3. **Field name in vLLM 0.28.0 is `reasoning`, not `reasoning_content`.**
   The OpenAI-style API and the streaming API both use the new name.

4. **First-start compile cache is large.** The torch.compile cache for
   the 1.7B model + 51 cudagraph sizes takes ~1 GiB on disk under
   `~/.cache/vllm/torch_compile_cache/`. Don't be alarmed by the 51s
   cudagraph capture step on first run.

5. **Drop `--enforce-eager` AND raise `VLLM_ENGINE_READY_TIMEOUT_S` for
   safety.** The env var exists (default 600s) but is irrelevant to the
   "Failed core proc(s): {}" failure path (that fires immediately on a
   spurious event, not after a timeout). Still set it to 1200s as
   headroom.

6. **YaRN context extension is via `--hf-overrides`, NOT `--rope-scaling`.**
   vLLM 0.28.0 rejects `--rope-scaling {...}` with "unrecognized
   arguments". The correct flag is `--hf-overrides
   '{"rope_scaling":{...}}'` (nested JSON). Documented in
   `scripts/02b_start_vllm_detached.sh`.

7. **254K context is 32.6 GiB, not 97 GiB.** The original 0.8 gpu-mem-util
   with 8K max-model-len was wildly over-allocated (vLLM pre-allocates
   a pool sized for max_num_seqs x max_model_len). For long-context
   lean profiles, set gpu-mem-util proportional to the actual
   KV-cache requirement: 112 KB/token x max_model_len, plus 3.4 GB
   model. See the table in the "YaRN context extension" section above.

## Next steps / TODO

- Try `Qwen/Qwen3-4B` (4B) and `Qwen/Qwen3-8B` (8B) for higher quality.
- Try a MoE thinking model (`nvidia/Qwen3.6-35B-A3B-NVFP4`) — the NVIDIA
  recipe's recommended agent-ready model for DGX Spark.
- Add a streaming chat-completions test (`stream: true`) to verify the
  reasoning parser works in stream mode (the field name should still
  be `reasoning`).
- Add an OpenAI Python client smoke test (the recipe's canonical
  client test).

## File map

- [README.md](README.md) — top-level doc
- [scripts/env.sh](scripts/env.sh) — config (model, port, GPU mem)
- [scripts/02b_start_vllm_detached.sh](scripts/02b_start_vllm_detached.sh) — canonical launcher
- [scripts/05_test_thinking.sh](scripts/05_test_thinking.sh) — think/content split test
- [scripts/04_logs.sh](scripts/04_logs.sh) — `docker logs -f` wrapper
- [.agent/README-vllm-dgx.md](.agent/README-vllm-dgx.md) — deep domain manual
- [outputs/](outputs/) — test artifacts, dated subfolders

## OpenClaw registration fix (2026-09-05)

`06_register_openclaw.sh` (v1) added the provider but the model was NOT
selectable in OpenClaw. Two root causes, both proven:

1. **Missing allowlist entry** - in this OpenClaw build the model picker only
   shows models listed in `agents.defaults.modelPolicy.allow`. The v1 script
   never touched it.
2. **Model id mismatch** - provider id was `qwen3-1.7b` but vLLM serves the
   HF handle `Qwen/Qwen3-1.7B` (no `--served-model-name` in the docker run).
   Proof: `POST /v1/chat/completions {model: "qwen3-1.7b"}` -> HTTP 404
   "model does not exist"; with `Qwen/Qwen3-1.7B` -> HTTP 200 + split
   `reasoning`/`content`.

Fixes applied to the live config (backup `openclaw.json.bak-vllmfix-20260905`):
provider id corrected, 5 stale per-agent keys renamed, ref added to the
allowlist. Hot reload confirmed 09:59:56. `/model` ref:

    /model local-vllm/Qwen/Qwen3-1.7B

`scripts/add_vllm_provider.py` updated to v2 (correct id + allowlist step) so
a re-run produces a working registration. Open point: confirm OpenClaw's
`openai-completions` client reads the v0.28 `reasoning` field (renamed from
`reasoning_content`); if the think block is dropped in chat, align the parser
or the client param mapping.
