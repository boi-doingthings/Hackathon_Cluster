#!/bin/bash -l
#SBATCH --job-name=humana-nemo-sdg
#SBATCH --partition=primary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm-humana-%j.out

set -euo pipefail
umask 0007
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/gpu-${SLURM_JOB_ID}}"
TEAM="${IAG_TEAM:-$(id -Gn | tr ' ' '\n' | grep -E '^iag-team[0-9]+$' | head -n 1 || true)}"
if [[ -z "$TEAM" ]]; then
  printf 'Could not detect IAG team; export IAG_TEAM before submission.\n' >&2
  exit 1
fi
TEAM_SCRATCH="${TEAM_SCRATCH:-/lustre/fs01/hackathons/teams/${TEAM}}"
RUNTIME_DIR="${TEAM_SCRATCH}/.iag/${USER:-user}"
VENV_DIR="${VENV_DIR:-${RUNTIME_DIR}/venvs/humana-grounded-sdg}"
mkdir -p "$RUNTIME_DIR/venvs" "$RUNTIME_DIR/cache/uv" "$OUTPUT_DIR"
export UV_CACHE_DIR="${RUNTIME_DIR}/cache/uv"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
module load uv
cd "$PROJECT_DIR"
uv sync --extra curator --extra dev
printf 'Host: %s\n' "$(hostname -f 2>/dev/null || hostname)"
printf 'Job: %s\n' "$SLURM_JOB_ID"
printf 'Project: %s\n' "$PROJECT_DIR"
printf 'Output: %s\n' "$OUTPUT_DIR"
nvidia-smi
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the allocated GPU job")
x = torch.randn((2048, 2048), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("gpu_matmul", tuple(y.shape), y.device)
PY
uv run python -m pytest -q
uv run ruff check src tests
args=(
  --engine nemo
  --records-per-chunk 2
  --support-conversations "${SUPPORT_CONVERSATIONS:-1200}"
  --support-seed "${SUPPORT_SEED:-20260722}"
  --workspace "$OUTPUT_DIR"
)
if [[ -n "${SOURCE_LIMIT:-}" ]]; then args+=(--limit "$SOURCE_LIMIT"); fi
uv run humana-sdg all "${args[@]}"
uv run python - "$OUTPUT_DIR" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
root = Path(sys.argv[1])
lines = []
for path in sorted(item for item in root.rglob('*') if item.is_file()):
    digest = sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
(root / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"checksummed_files={len(lines)}")
PY
printf 'PIPELINE_COMPLETE output=%s\n' "$OUTPUT_DIR"
