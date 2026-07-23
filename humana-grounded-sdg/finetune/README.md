# Humana Customer Support SLM — Fine-tuning

NeMo Evaluator / Safe Synthesizer SQS-DPS remain **blocked** without NMP credentials.
This package fine-tunes a small instruct model with **LoRA SFT** on the 1,200 grounded support transcripts.

## Data

| Split | Rows | Path |
|------:|-----:|------|
| Train | 1140 | `data/support_sft_train.jsonl` |
| Val | 60 | `data/support_sft_val.jsonl` |
| Source | 1200 | `../results/humana-support-slm-1200/support_sft_openai.jsonl` |

Format: OpenAI chat `messages` (8 turns: user/assistant alternating).

## Local smoke (CPU OK)

```bash
cd humana-grounded-sdg/finetune
python -m pip install -r requirements.txt
python scripts/train_lora_sft.py --config configs/lora_smollm_smoke.yaml
python scripts/generate_eval_samples.py \
  --base HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter outputs/smoke-smollm135-lora/adapter \
  --out outputs/smoke-smollm135-lora/sample_generations.json
```

## GPU cluster (H100)

From `humana-grounded-sdg`:

```bash
sbatch finetune/cluster/run_finetune_lora.sbatch
```

Uses `configs/lora_qwen05b_gpu.yaml` (Qwen2.5-0.5B-Instruct + LoRA).

## Safety notes

- Training data is **synthetic / public-doc grounded** only.
- Model must not be treated as a coverage decision engine.
- Keep human-verification behavior in prompts/evals.
