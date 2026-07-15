#!/bin/bash -l
#SBATCH --job-name=iag-gpu-smoke
#SBATCH --partition=primary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

echo "Host: $(hostname -f 2>/dev/null || hostname)"
echo "Job id: ${SLURM_JOB_ID}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not set}"
echo

module load rootless-docker
if command -v start_rootless_docker >/dev/null 2>&1; then
    start_rootless_docker
else
    start_rootless_docker.sh
fi
docker version
echo

nvidia-smi
echo
nvidia-smi -L

python3 - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
except Exception as exc:
    print("torch check skipped or failed:", exc)
PY
