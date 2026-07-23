#!/usr/bin/env python
"""Generate a few support answers from base vs LoRA adapter for qualitative check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "I need to understand the documented Medicare appeal process. What is the first appeal level?",
    "Where can I verify Part D pharmacy coverage rules from public plan documents?",
    "Can you approve my claim payment right now based on public guidance?",
]


def gen(model, tok, prompt: str, max_new_tokens: int = 180) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(gen_ids, skip_special_tokens=True)


def load_model(base: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(adapter or base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    model, tok = load_model(args.base, args.adapter)
    for p in PROMPTS:
        ans = gen(model, tok, p)
        rows.append({"prompt": p, "completion": ans, "adapter": args.adapter, "base": args.base})
        print("Q:", p)
        print("A:", ans[:400])
        print("---")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
