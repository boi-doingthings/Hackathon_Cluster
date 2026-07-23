#!/usr/bin/env python
"""LoRA SFT for Humana customer-support SLM from OpenAI-chat JSONL."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    train_file = Path(cfg["train_file"])
    val_file = Path(cfg["val_file"])
    if not train_file.is_absolute():
        train_file = root / train_file
    if not val_file.is_absolute():
        val_file = root / val_file

    out_dir = Path(args.output_dir or cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model_name_or_path"]
    print(f"[finetune] model={model_name}")
    print(f"[finetune] train={train_file} val={val_file}")
    print(f"[finetune] out={out_dir}")
    print(f"[finetune] cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float32
    if cfg.get("bf16") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif cfg.get("fp16") and torch.cuda.is_available():
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    ds = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )

    def to_text(example):
        messages = example["messages"]
        # Prefer chat template when available
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            parts = []
            for m in messages:
                parts.append(f"{m.get('role','').upper()}: {m.get('content','')}")
            text = "\n".join(parts)
        return {"text": text}

    ds = ds.map(to_text, remove_columns=[c for c in ds["train"].column_names if c != "text"])

    lora = LoraConfig(
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.get("lora_target_modules") or ["q_proj", "v_proj"]),
    )

    max_steps = int(cfg.get("max_steps", -1))
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=max_steps,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        eval_strategy="steps" if int(cfg.get("eval_steps", 0) or 0) > 0 else "no",
        eval_steps=int(cfg.get("eval_steps", 50) or 50),
        save_steps=int(cfg.get("save_steps", 50) or 50),
        save_total_limit=2,
        bf16=bool(cfg.get("bf16") and torch.cuda.is_available()),
        fp16=bool(cfg.get("fp16") and torch.cuda.is_available()),
        seed=int(cfg.get("seed", 42)),
        report_to=cfg.get("report_to", "none"),
        max_length=int(cfg.get("max_seq_length", 1024)),
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if sft_args.eval_strategy != "no" else None,
        processing_class=tok,
        peft_config=lora,
    )

    train_result = trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))

    metrics = {
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_loss": train_result.metrics.get("train_loss"),
        "train_samples": len(ds["train"]),
        "val_samples": len(ds["validation"]),
        "model_name_or_path": model_name,
        "output_dir": str(out_dir),
        "cuda": torch.cuda.is_available(),
        "max_steps": max_steps,
        "config": cfg,
    }
    # optional eval
    if sft_args.eval_strategy != "no":
        ev = trainer.evaluate()
        metrics["eval"] = ev

    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("[finetune] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
