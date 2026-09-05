#!/usr/bin/env bash
# vLLM environment configuration for DGX Spark (GB10).
# Source this file from other scripts: source /home/jagones/Programs/vLLM/scripts/env.sh

# HF token loaded from openclaw gateway env (single source of truth).
HF_TOKEN=$(grep '^HF_TOKEN=' /home/jagones/.openclaw/gateway.systemd.env | head -1 | cut -d= -f2-)
export HF_TOKEN="${HF_TOKEN}"

# Small thinking model: Qwen3-1.7B is the smallest Qwen3 family with native thinking.
# Other options: Qwen/Qwen3-4B (4B), Qwen/Qwen3-8B, microsoft/Phi-4-mini-reasoning.
MODEL_HANDLE="Qwen/Qwen3-1.7B"
export MODEL_HANDLE

# vLLM image (NVIDIA recipe: containerized vLLM, supports GB10/SM_121).
VLLM_IMAGE="vllm/vllm-openai:latest"
export VLLM_IMAGE

# Server port (avoid llama-server's 8130).
VLLM_PORT=8000
export VLLM_PORT

# Max context (prompt+output). Qwen3-1.7B has max_position_embeddings=40960 (40K
# native). YaRN rope scaling extends to 254K. KV cache at 254K = ~28 GB BF16,
# plus 3.4 GB model + 3 GB overhead = ~35 GB. Set GPU_MEM_UTIL=0.28 to match.
MAX_MODEL_LEN=254000
export MAX_MODEL_LEN

# YaRN rope scaling (Qwen3-1.7B native 40K -> 254K = factor 6.4).
# Passed to vLLM as --rope-scaling. Empty string disables scaling (40K max).
ROPE_SCALING='{"rope_type":"yarn","factor":6.4,"original_max_position_embeddings":40960}'
export ROPE_SCALING

# GPU memory fraction for weights+KV cache.
# 0.28 = ~35 GB on the 128 GB DGX, enough for the 1.7B model + 254K KV cache
# + headroom for activations and CUDA workspaces. Set to 0.18 for 131K (factor 3.2)
# or 0.12 for 65K (factor 1.6).
GPU_MEM_UTIL=0.28
export GPU_MEM_UTIL

# Max tokens processed per batch. Bumped from 8192 -> 32768 so a single 254K
# request can make real progress per iteration (otherwise each step only
# processes 8K and generation crawls).
MAX_NUM_BATCHED_TOKENS=32768
export MAX_NUM_BATCHED_TOKENS

# Container name.
CONTAINER_NAME="vllm-server"
export CONTAINER_NAME

# HF cache mount (DGX home: /home/jagones).
HF_CACHE_HOST="/home/jagones/.cache/huggingface"
export HF_CACHE_HOST
