#!/usr/bin/env bash
set -euo pipefail

: "${DYNAOD_LORA_CONFIG:=configs/qwen2.5_1.5b_lora_train.yaml}"

llamafactory-cli train "$DYNAOD_LORA_CONFIG"
