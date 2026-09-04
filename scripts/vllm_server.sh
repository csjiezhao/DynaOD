#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${MODEL_NAME:=qwen2.5-1.5b-local}"
: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen checkpoint or merged SFT model path.}"
: "${vLLM_API_KEY:?Set vLLM_API_KEY for the OpenAI-compatible vLLM server.}"
: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8000}"

export CUDA_VISIBLE_DEVICES

exec python -m vllm.entrypoints.openai.api_server \
  --served-model-name "$MODEL_NAME" \
  --api-key "$vLLM_API_KEY" \
  --model "$MODEL_PATH" \
  --trust-remote-code \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --max-model-len 4096 \
  --disable-log-stats \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95
