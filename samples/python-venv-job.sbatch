#!/bin/bash -l
#SBATCH --job-name=iag-python-venv
#SBATCH --partition=primary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
umask 0007

TEAM="${IAG_TEAM:-}"
if [[ -z "$TEAM" ]]; then
    TEAM="$(id -Gn | tr ' ' '\n' | grep -E '^iag-team[0-9]+$' | head -n 1 || true)"
fi

if [[ -z "$TEAM" ]]; then
    echo "Could not detect team. Set IAG_TEAM=iag-team<N> before submitting." >&2
    exit 1
fi

TEAM_SCRATCH="${TEAM_SCRATCH:-${HOME}/${TEAM}}"
PROJECT_DIR="${PROJECT_DIR:-${TEAM_SCRATCH}/example-python-job}"
RUNTIME_DIR="${TEAM_SCRATCH}/.${USER:-user}"
VENV_DIR="${VENV_DIR:-${RUNTIME_DIR}/venvs/example-python}"

mkdir -p "$RUNTIME_DIR/venvs" "$RUNTIME_DIR/cache/uv" "$RUNTIME_DIR/cache/pip"
chmod 700 "$RUNTIME_DIR" 2>/dev/null || true
export UV_CACHE_DIR="${RUNTIME_DIR}/cache/uv"
export PIP_CACHE_DIR="${RUNTIME_DIR}/cache/pip"

docker info >/dev/null
docker version
echo

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load "${IAG_UV_MODULE:-py-uv/0.6.8}" >/dev/null 2>&1 || module load uv >/dev/null 2>&1 || true
fi

if [[ ! -d "$VENV_DIR" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv "$VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
    fi
fi

source "$VENV_DIR/bin/activate"
if command -v uv >/dev/null 2>&1; then
    echo "Using $(uv --version)"
else
    echo "Using $(python --version) with pip"
    python -m ensurepip --upgrade >/dev/null 2>&1 || true
fi

# Replace this with your project requirements.
if command -v uv >/dev/null 2>&1; then
    uv pip install torch
else
    python -m pip install torch
fi

echo "Host: $(hostname -f 2>/dev/null || hostname)"
echo "Project: $PROJECT_DIR"
echo "Venv: $VENV_DIR"
echo

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("gpu matmul ok:", tuple(y.shape))
PY
