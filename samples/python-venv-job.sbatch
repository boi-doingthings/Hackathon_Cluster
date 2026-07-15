#!/bin/bash -l
#SBATCH --job-name=iag-python-venv
#SBATCH --partition=primary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

TEAM="${IAG_TEAM:-}"
if [[ -z "$TEAM" ]]; then
    TEAM="$(id -Gn | tr ' ' '\n' | grep -E '^iag-team[0-9]+$' | head -n 1 || true)"
fi

if [[ -z "$TEAM" ]]; then
    echo "Could not detect team. Set IAG_TEAM=iag-team<N> before submitting." >&2
    exit 1
fi

TEAM_SCRATCH="${TEAM_SCRATCH:-/lustre/fs01/hackathons/teams/${TEAM}}"
PROJECT_DIR="${PROJECT_DIR:-${TEAM_SCRATCH}/example-python-job}"
VENV_DIR="${VENV_DIR:-${TEAM_SCRATCH}/venvs/example-python}"

module load rootless-docker
if command -v start_rootless_docker >/dev/null 2>&1; then
    start_rootless_docker
else
    start_rootless_docker.sh
fi
docker version
echo

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    module load uv
    uv venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
module load uv
uv pip install --upgrade pip

# Replace this with your project requirements.
uv pip install torch

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
