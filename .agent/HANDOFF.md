# HANDOFF — vLLM on DGX Spark

Last updated: 2026-09-05

## Current state

**Status: WORKING.** vLLM 0.28.0 (`vllm/vllm-openai:latest`) serves
`Qwen/Qwen3-1.7B` on `localhost:8000` on the DGX Spark. The reasoning
parser splits `message.reasoning` (think) from `message.content`
(answer). Confirmed end-to-end with two test prompts on 2026-09-05.

## Server

- **Image:** `vllm/vllm-openai:latest` (v0.28.0, 20.6 GB on disk)
- **Model:** `Qwen/Qwen3-1.7B` (3.78 GiB checkpoint, BF16)
- **Port:** 8000 (host) -> 8000 (container)
- **Container name:** `vllm-server`
- **GPU memory utilization:** 0.8 (NVIDIA recipe-recommended)
- **Max model len:** 8192 tokens
- **First-start time:** ~130s end-to-end (safetensors load 20s + torch.compile
  cudagraph capture 51s + FlashInfer autotune 18s + warmup + handshake).
  Subsequent starts are much faster due to on-disk compile cache.

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
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
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
