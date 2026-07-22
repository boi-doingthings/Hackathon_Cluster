#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv sync --extra dev
rm -rf outputs/smoke
uv run humana-sdg all --engine python --limit 1 --records-per-chunk 2 --conversation-records 1000 --min-conversation-use-cases 1 --workspace outputs/smoke
uv run python -m pytest -q
uv run ruff check src tests
printf 'Offline smoke output: %s\n' "$PWD/outputs/smoke"
