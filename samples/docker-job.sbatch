#!/bin/bash -l
#SBATCH --job-name=iag-docker
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
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.12-py3}"

cd "$TEAM_SCRATCH"

docker info >/dev/null

echo "Host: $(hostname -f 2>/dev/null || hostname)"
echo "Image: $IMAGE"
echo "Team storage: $TEAM_SCRATCH"
echo

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${TEAM_SCRATCH}:${TEAM_SCRATCH}" \
    -w "$TEAM_SCRATCH" \
    "$IMAGE" \
    python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
PY
