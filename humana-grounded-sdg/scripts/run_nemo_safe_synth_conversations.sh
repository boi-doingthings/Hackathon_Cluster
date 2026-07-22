#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${NMP_BASE_URL:?Set NMP_BASE_URL to a deployed NeMo Platform endpoint}"
: "${NMP_WORKSPACE:?Set NMP_WORKSPACE to the authorized workspace}"
: "${NMP_ACCESS_TOKEN:?Set NMP_ACCESS_TOKEN without writing it to disk}"

CONVERSATIONS="${CONVERSATIONS:-outputs/run/synthetic/customer_support_transcripts.jsonl}"
OUTPUT="${SAFE_SYNTH_OUTPUT:-outputs/safe_synth_conversations}"
if [[ ! -f "$CONVERSATIONS" ]]; then
  printf 'Conversation dataset not found: %s\n' "$CONVERSATIONS" >&2
  exit 2
fi

uv sync --extra platform
uv run humana-sdg safe-synthesize-conversations \
  --conversations "$CONVERSATIONS" \
  --output "$OUTPUT"
printf 'Safe Synthesizer artifacts: %s\n' "$OUTPUT"
