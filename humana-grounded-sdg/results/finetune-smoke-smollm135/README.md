# Fine-tune smoke result

- Base: HuggingFaceTB/SmolLM2-135M-Instruct
- Method: LoRA SFT (r=8, alpha=16)
- Steps: 3 (CPU smoke)
- Train subset: 64 / Val subset: 16
- Full data ready: train 1140 / val 60 in `finetune/data/`
- train_loss: see train_metrics.json
- NeMo Evaluator / official SQS-DPS: blocked (no NMP)
- GPU full train: `sbatch finetune/cluster/run_finetune_lora.sbatch`
