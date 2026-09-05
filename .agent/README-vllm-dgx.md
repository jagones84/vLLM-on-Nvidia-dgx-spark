# vLLM on DGX Spark — Domain Manual

Frozen, tested, working architecture. Last updated 2026-09-05.
DGX Spark: GB10, sm_121, 128 GB unified LPDDR5x, driver 580.173.02, CUDA 13.0.

## Golden rules

1. **Do NOT pass `--enforce-eager`** with Qwen3 + sm_121. The
   `RuntimeError: Engine core initialization failed. ... Failed core
   proc(s): {}` failure is *not* a timeout — it's a poller event firing
   under eager mode that is harmless under torch.compile. Fix: let
   torch.compile run, expect ~130s first start.

2. **Reasoning-parser naming.** Qwen3 family -> `qwen3`. DeepSeek-R1
   family -> `deepseek_r1`. Generic -> omit (the model will just emit
   think blocks inline in `content`).

3. **Field name in the response.** vLLM 0.28.0 renamed
   `reasoning_content` -> `reasoning`. Clients that still read
   `reasoning_content` get `null`. Always read both in test scripts.

4. **GPU memory budget.** Start at 0.8 (NVIDIA recommendation for shared
   GPUs). 0.85+ can work but the cudagraph capture may OOM on first
   start with 51 capture sizes + 1.1 GiB graph memory.

5. **Watch `usage.completion_tokens_details.reasoning_tokens`.** If this
   is near `max_tokens` and `content` is empty / null, the model ran out
   of budget mid-think. Either lower the budget's expected answer length
   or raise `max_tokens` (default 4096 in the test).

6. **Container name = `vllm-server`.** `02b_start_vllm_detached.sh`
   removes any pre-existing container with the same name before launching.

7. **HF token lives in `~/.openclaw/gateway.systemd.env`.** The `env.sh`
   script greps it out and exports `HF_TOKEN`. Do not hardcode tokens in
   scripts.

## Canonical launch (Qwen3-1.7B)

```bash
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "" \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1200 \
  -v /home/jagones/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
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

## Test (think/content split)

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"}],
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "chat_template_kwargs": {"enable_thinking": true}
  }' | python3 -m json.tool
```

Expected:
- `choices[0].message.reasoning`: long step-by-step think block
- `choices[0].message.content`: the final answer
- `choices[0].finish_reason`: "stop"
- `usage.completion_tokens_details.reasoning_tokens`: > 0

## Diagnosis cheatsheet

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: Engine core initialization failed. ... Failed core proc(s): {}` ~1s after `init engine took N s` | `--enforce-eager` under Qwen3 + sm_121 | Remove `--enforce-eager` |
| `flashinfer.jit: [Autotuner]: Saved 0 configs ...` | sm_121 has 48 SMs, less than torch's max_autotune threshold; harmless | Ignore |
| `Not enough SMs to use max_autotune_gemm mode` | Same as above | Ignore |
| Server takes 90-130s to first health | torch.compile + cudagraph capture | Expected. Don't kill it. |
| `finish_reason: length` with `content: null` | `max_tokens` too low; model spent budget on thinking | Raise `max_tokens` to 4096+ |
| `connection refused` on 8000 | Container never started | `docker logs vllm-server` |
| `ReasoningParserNotFoundError: deepseek_r1` | Wrong parser name for Qwen3 | Use `qwen3` |
| OOM in cudagraph capture | `gpu-memory-utilization` too high | Drop to 0.8 or lower |

## Rollback / cleanup

```bash
# Stop and remove container
docker stop vllm-server && docker rm vllm-server

# Remove the compile cache (forces re-compile on next start)
rm -rf /home/jagones/.cache/vllm/torch_compile_cache
rm -rf /home/jagones/.cache/vllm/flashinfer_autotune_cache

# Remove the model
rm -rf /home/jagones/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B

# Remove the vLLM image
docker rmi vllm/vllm-openai:latest
```

## Tested models

| Model | Status | Notes |
|---|---|---|
| `Qwen/Qwen3-1.7B` | ✅ works | Default in `env.sh`. 3.78 GiB BF16 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ✅ works (debug only) | Not a thinking model; used to prove the pipeline before debugging Qwen3 |

## Tested parser + tool combos

| Combination | Status |
|---|---|
| `--reasoning-parser qwen3` + `--tool-call-parser hermes` | ✅ |
| `--reasoning-parser deepseek_r1` on Qwen3 | ❌ wrong parser |
| no reasoning-parser, thinking model | think block ends up inline in `content` |
| `--enforce-eager` + Qwen3 | ❌ engine init fails |

## References

- https://build.nvidia.com/spark/vllm/instructions
- https://recipes.vllm.ai/Qwen/Qwen3-1.7B
- https://build.nvidia.com/spark/vllm/agent-ready-models
- vLLM source: `vllm/v1/engine/utils.py:1286` (wait_for_engine_startup)
- vLLM source: `vllm/envs.py:25` (VLLM_ENGINE_READY_TIMEOUT_S, default 600)
