#!/bin/bash -l
#SBATCH --job-name=humana-full-eval
#SBATCH --partition=primary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-humana-eval-%j.out

set -euo pipefail
: "${EVAL_WORKSPACE:?Set EVAL_WORKSPACE to an existing pipeline output directory}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
TEAM="${IAG_TEAM:-$(id -Gn | tr ' ' '\n' | grep -E '^iag-team[0-9]+$' | head -n 1 || true)}"
TEAM_SCRATCH="${TEAM_SCRATCH:-/lustre/fs01/hackathons/teams/${TEAM}}"
RUNTIME_DIR="${TEAM_SCRATCH}/.iag/${USER:-user}"
export UV_CACHE_DIR="${RUNTIME_DIR}/cache/uv"
export UV_PROJECT_ENVIRONMENT="${RUNTIME_DIR}/venvs/humana-grounded-sdg"
module load uv
cd "$PROJECT_DIR"
uv sync --extra curator --extra dev
nvidia-smi
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY
uv run python -m pytest -q
uv run ruff check src tests
uv run humana-sdg evaluate \
  --records "$EVAL_WORKSPACE/synthetic/grounded_synthetic.jsonl" \
  --curated "$EVAL_WORKSPACE/curated/chunks.jsonl" \
  --output "$EVAL_WORKSPACE/evaluation.json"
uv run python - "$EVAL_WORKSPACE" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
root = Path(sys.argv[1])
paths = sorted(item for item in root.rglob('*') if item.is_file() and item.name != 'manifest.sha256')
lines = [f"{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}" for path in paths]
(root / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"checksummed_files={len(lines)}")
PY
printf 'EVALUATION_COMPLETE output=%s\n' "$EVAL_WORKSPACE"
