#!/bin/bash
#SBATCH --job-name=humana-sft-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/outputs/slurm-%j.out
#SBATCH --error=finetune/outputs/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# repo root may be Hackathon_Cluster or humana-grounded-sdg
if [ -d humana-grounded-sdg ]; then cd humana-grounded-sdg; fi
if [ -f ../pyproject.toml ] && [ -d data ]; then cd ..; fi
cd "$(dirname "$0")/.."

module load uv 2>/dev/null || true
export UV_HTTP_TIMEOUT=300
uv venv .venv-ft || true
# shellcheck disable=SC1091
source .venv-ft/bin/activate
uv pip install -r requirements.txt
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu124

python scripts/train_lora_sft.py --config configs/lora_qwen05b_gpu.yaml
python scripts/generate_eval_samples.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/gpu-qwen05b-lora/adapter \
  --out outputs/gpu-qwen05b-lora/sample_generations.json

echo "__FINETUNE_DONE__"
