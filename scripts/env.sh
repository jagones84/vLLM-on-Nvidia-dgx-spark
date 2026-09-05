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

# Max context (prompt+output). NVIDIA DGX Spark + 1.7B model can go bigger,
# but 8192 keeps KV cache modest and startup reliable.
MAX_MODEL_LEN=8192
export MAX_MODEL_LEN

# GPU memory fraction for weights+KV cache. NVIDIA recipe recommends 0.8.
GPU_MEM_UTIL=0.8
export GPU_MEM_UTIL

# Container name.
CONTAINER_NAME="vllm-server"
export CONTAINER_NAME

# HF cache mount (DGX home: /home/jagones).
HF_CACHE_HOST="/home/jagones/.cache/huggingface"
export HF_CACHE_HOST
