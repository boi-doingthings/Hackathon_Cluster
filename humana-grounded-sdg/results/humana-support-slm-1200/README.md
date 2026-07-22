# Humana Support SLM Synthetic Dataset (1,200 Conversations)

This artifact contains 1,200 deterministic, fully synthetic, eight-turn customer-support conversations grounded in public Humana/CMS documents. It contains no real member data.

## Files

- `support_conversations.jsonl`: governed transcript schema with citations and safety metadata.
- `support_sft_openai.jsonl`: alternating `user`/`assistant` SFT messages.
- `verification_summary.json`: counts, distributions, local quality results, deterministic replay check, and NeMo status.
- `support_evaluation.json`: fail-closed local grounding/privacy/safety evaluation.
- `grounded_record_evaluation.json`: companion grounded-record evaluation.
- `source_manifest.json` and `download_receipts.json`: public-source provenance.
- `nemo_safe_synth_attempt.json`: authentic Safe Synthesizer attempt evidence; no official SQS/DPS were returned because no NeMo Platform service was reachable.
- `nemo_evaluator_attempt.json`: authentic NeMo Evaluator attempt evidence; the launcher/API key were unavailable.
- `manifest.sha256`: SHA-256 checksums for every artifact file.

## Verified H100 / Slurm run

Slurm job `13526` completed on `gpu004` with exit `0:0` using an NVIDIA H100 80GB HBM3 GPU. CUDA 12.4, PyTorch 2.6.0+cu124, GPU matrix multiplication, tests, Ruff, the NeMo Curator pipeline, 6,462 grounded records, and 1,200 support conversations all completed successfully. The GPU JSONL hashes match the local release files exactly. See `gpu_run_receipt.json`, `gpu_job_13526.log`, and `gpu_manifest.sha256`.

## Score provenance

The local evaluator passed 100% citation validity, grounded quotes, turn structure, and safety compliance with zero personal-PII findings and zero exact duplicates. These local metrics are **not** NeMo SQS/DPS or NeMo Evaluator scores. Official NeMo fields remain `null` and fail closed; see the attempt evidence files.
