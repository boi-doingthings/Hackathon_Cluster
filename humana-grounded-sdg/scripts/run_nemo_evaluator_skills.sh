#!/usr/bin/env bash
set -euo pipefail

: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY without writing it into this repository}"
export NEMO_MODEL_URL="${NEMO_MODEL_URL:-https://integrate.api.nvidia.com/v1/chat/completions}"
export NEMO_MODEL_ID="${NEMO_MODEL_ID:-nvidia/nemotron-3-super-120b-a12b}"
MAX_PROBLEMS="${MAX_PROBLEMS:-20}"

command -v nel >/dev/null || {
  printf 'Install NeMo Evaluator with Skills support first.\n' >&2
  exit 1
}
nel eval run --bench skills://bfcl_v4 --repeats 1 --max-problems "$MAX_PROBLEMS"
