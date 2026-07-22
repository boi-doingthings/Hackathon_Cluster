# Humana Grounded Synthetic Data with NVIDIA NeMo

A reproducible, citation-preserving pipeline for producing **synthetic** healthcare-insurance training data from public Humana and US-government documents. The production path runs **NeMo Curator first**, then prepares grounded QA, tool-calling and embedding-customization datasets, applies fail-closed quality/privacy gates, and optionally submits jobs to NeMo Platform services.

This repository does **not** contain member records, PHI, copied PDFs, credentials, or an assertion that synthetic output is an official Humana coverage decision.

## What is implemented

1. Download 22 verified public PDFs from official `humana.com`, `cms.gov`, and `medicare.gov` hosts.
2. Validate HTTPS host, status, MIME type, PDF magic bytes, file size and SHA-256.
3. Extract page-level text with source URL, publisher, year, page number and document hash.
4. Run NeMo Curator `JsonlReader -> AddId -> WordCountFilter -> JsonlWriter` as the first production curation stage.
5. Apply deterministic exact-text deduplication without losing page citations.
6. Generate page-grounded synthetic QA/explanation examples with `chunk_id` citations.
7. Generate 1,200 deterministic, multi-turn customer-support transcripts across 12 use cases, plus OpenAI SFT JSONL.
8. Generate OpenAI-format, single-call insurance tool-use examples using `SYN-*` identifiers only.
9. Generate `query` / `pos_doc` / `neg_doc` triplets for NeMo embedding customization.
10. Gate output on citation validity, quote grounding, answer support, duplicates and sensitive PII patterns.
11. Optionally run NeMo Safe Synthesizer, Data Designer, Customizer and NeMo Evaluator Skills.
12. Run in a Slurm GPU allocation with a verified CUDA matrix multiplication and auditable SHA-256 manifest.

## Safety and lawful-use boundary

- The Humana PDFs remain subject to their publisher's terms. This project stores only links and downloads files at runtime; review Humana terms before redistribution or commercial reuse.
- US-government documents are public resources, but agency marks and third-party material can have separate restrictions.
- Do not add claims, EHR, call-center, enrollment or member files unless your organization has authorization, a documented lawful basis, de-identification review and security controls.
- Generated records use explicit synthetic labels and must not be used for eligibility, payment, medical advice, prior-authorization or coverage decisions.
- The generator must not invent citations. A record is blocked unless its quote resolves to the exact source chunk and page.
- Synthetic tool examples use `SYN-*` references; they never call a real payer system.

## Architecture

```text
corpus/manifest.json
        |
        v
allowlisted downloader --> PDF magic/MIME/SHA-256 receipts
        |
        v
page-aware PyMuPDF extraction
        |
        v
NeMo Curator (required production first stage)
JsonlReader -> AddId -> WordCountFilter -> JsonlWriter
        |
        v
exact dedup + governed chunks
        |
        +--> grounded_synthetic.jsonl --> custom grounding/privacy evaluation
        |
        +--> support_conversations.jsonl --> support SLM training + local quality gates
        |                                --> support_sft_openai.jsonl
        |
        +--> tool_calling_openai.jsonl --> Data Designer / Customizer / BFCL v4
        |
        +--> embedding_triplets/{training,validation}.jsonl --> embedding Customizer
        |
        +--> Safe Synthesizer --> synthetic CSV + SQS/DPS HTML report
```

A raw GPU machine and a NeMo Platform deployment are different things. The local Curator and deterministic pipeline can run on the supplied Slurm cluster. Safe Synthesizer, Data Designer, Customizer and managed Evaluator calls require a reachable NeMo Platform deployment with its services, models, secrets and authentication already configured.

## Quick start: verified offline smoke

The offline smoke is deterministic and uses the Python fallback only so it can run on Windows without NeMo Curator. It is a test path, not the required production curation path.

```bash
cd humana-grounded-sdg
bash scripts/run_offline_smoke.sh
```

It downloads one official PDF, extracts and curates it, creates all three dataset formats, evaluates them, runs tests and runs Ruff.

Expected success marker:

```text
"evaluation_passed": true
Offline smoke output: .../outputs/smoke
```

## Production NeMo Curator run

Linux and Python 3.11 are recommended.

```bash
uv sync --extra curator --extra dev
uv run humana-sdg all \
  --engine nemo \
  --workspace outputs/full
```

To run a small production-path check:

```bash
uv run humana-sdg all \
  --engine nemo \
  --limit 1 \
  --workspace outputs/nemo-smoke
```

`--engine nemo` is the default. If `nemo-curator` is unavailable, the program fails instead of silently falling back.

## Slurm GPU run

From the cluster login node, clone this repository into team storage and submit from this project directory:

```bash
export IAG_TEAM=iag-team<N>
export TEAM_SCRATCH=/lustre/fs01/hackathons/teams/$IAG_TEAM
cd "$TEAM_SCRATCH/Hackathon_Cluster/humana-grounded-sdg"

# Optional: one-source end-to-end validation before the full corpus.
export SOURCE_LIMIT=1
sbatch cluster/run_humana_pipeline.sbatch
```

For the full 22-document run:

```bash
unset SOURCE_LIMIT
sbatch cluster/run_humana_pipeline.sbatch
```

The batch job:

- requests one GPU from `primary`;
- creates a reusable team-storage virtual environment;
- installs NeMo Curator 1.2.0;
- prints `nvidia-smi`;
- requires `torch.cuda.is_available()`;
- executes a CUDA matrix multiplication;
- runs tests and Ruff;
- runs the NeMo-first pipeline;
- writes `evaluation.json` and `manifest.sha256`.

### Verified full GPU run

The complete 22-PDF corpus was executed on the supplied cluster on 2026-07-22:

| Item | Verified result |
|---|---|
| Slurm job | `13372`, `COMPLETED`, exit `0:0`, elapsed `00:01:34` |
| Node / GPU | `gpu004`, NVIDIA H100 80GB HBM3 |
| Driver / CUDA / PyTorch | `550.90.07` / `12.4` / `2.6.0+cu124` |
| CUDA check | `torch.cuda.is_available() == True`; 2048x2048 matrix multiply on `cuda:0` |
| Tests / lint | 17 Pytest tests passed; Ruff passed |
| Sources | 22 PDFs, 46,071,807 bytes, each SHA-256 locked |
| Curated chunks | 3,231 |
| Grounded synthetic records | 6,462 |
| Tool-calling records | 200 |
| Embedding triplets | 2,746 training + 485 validation |
| Grounding / privacy gates | 100% valid citations, quotes and answer support; 0 duplicates; 0 personal-PII findings |

The run used commit `daa7d0b094af0e62e52b8f97611325b916c83599`. Audit evidence is committed under [`results/gpu-full-13372`](results/gpu-full-13372). The raw PDFs and generated datasets remain in team storage and are intentionally excluded from Git.

## CLI

```text
humana-sdg download
humana-sdg extract
humana-sdg curate                 # default engine: nemo
humana-sdg generate
humana-sdg evaluate
humana-sdg evaluate-support
humana-sdg all
humana-sdg safe-synthesize
humana-sdg safe-synthesize-conversations
humana-sdg data-designer-tools
humana-sdg customize-embeddings
```

Use `uv run humana-sdg COMMAND --help` for arguments.

## Output schemas

### Grounding chunk

Each JSONL row contains:

```json
{
  "chunk_id": "sha256...",
  "source_id": "humana_provider_delegated_2026",
  "source_title": "Humana Provider Manual (Delegated Providers), 2026",
  "source_url": "https://assets.humana.com/...",
  "source_sha256": "sha256...",
  "publisher": "Humana",
  "year": 2026,
  "category": "provider_operations",
  "page": 12,
  "text": "...",
  "citation_label": "Humana Provider Manual (Delegated Providers), 2026, p. 12"
}
```

### Grounded synthetic record

Every record contains an exact `chunk_id`, URL, page and quoted support:

```json
{
  "record_id": "sha256...",
  "task_type": "grounded_qa",
  "question": "...",
  "answer": "The cited passage states: ...",
  "citations": [
    {
      "chunk_id": "sha256...",
      "source_id": "...",
      "source_url": "https://...",
      "page": 12,
      "quote": "...",
      "citation_label": "..., p. 12"
    }
  ],
  "is_synthetic": true,
  "disclaimer": "Synthetic training example ..."
}
```

### Customer-support conversation

`support_conversations.jsonl` contains 1,200 deterministic eight-turn conversations across 12 Humana support use cases. Every item carries a synthetic-only case reference, channel, persona, outcome, exact page citation, public-source quote, explicit no-determination disclaimer and human-verification requirement. `support_sft_openai.jsonl` exports the same conversations as alternating `user` / `assistant` messages for SLM fine-tuning.

The fail-closed `humana-sdg evaluate-support` command requires 100% citation validity, grounded quotes, valid turn structure and safety compliance; zero personal-PII findings; at least 10 use cases; and no more than 2% exact duplicates.

### Tool-calling record

`tool_calling_openai.jsonl` follows the OpenAI `messages` + `tools` format used by the NVIDIA tool-calling tutorial. It allows one call per example and uses only `SYN-*` references.

### Embedding customization triplet

```json
{
  "query": "What guidance does ... provide about claims billing?",
  "pos_doc": "page-grounded positive chunk",
  "neg_doc": ["a distinct chunk from another source"]
}
```

## Evaluation gates

`humana-sdg evaluate` fails with exit code 2 unless all gates pass:

| Gate | Default |
|---|---:|
| Citation metadata resolves to a chunk | 100% |
| Citation quote appears in that exact chunk | 100% |
| Answer supported by citation quote | at least 95% |
| Exact duplicate rate | at most 5% |
| SSN, MBI-like ID, DOB or personal-email findings | 0 |

Public organizational contact addresses on allowlisted payer, Humana, federal and state-government domains are not treated as personal email addresses. This is not a replacement for enterprise DLP, privacy counsel or Safe Synthesizer's PII replay analysis.

## NeMo Safe Synthesizer

Prerequisites:

1. Deploy NeMo Platform 26.3-compatible Safe Synthesizer.
2. Authenticate (`nemo auth login --base-url <NMP_BASE_URL>` or provide an access token).
3. Ensure the configured classify/generation provider exists.
4. Put secrets in NeMo Platform's secret store, not in this repository.

```bash
uv sync --extra platform
export NMP_BASE_URL=https://your-nmp.example
export NMP_WORKSPACE=default
export NMP_MODEL_PROVIDER=system/nvidia-build
# Optional if your platform requires it:
export NMP_ACCESS_TOKEN=...
export NMP_HF_SECRET_NAME=hf-token

uv run humana-sdg safe-synthesize-conversations \
  --conversations data/synthetic/support_conversations.jsonl \
  --output outputs/safe_synth_conversations
```

The adapter follows the official builder sequence:

```text
with_data_source -> with_classify_model_provider -> with_replace_pii -> synthesize
```

It downloads:

- `safe_synthetic.csv`
- `safe_synth_evaluation.html`
- `safe_synth_summary.json` with SQS and DPS

Safe Synthesizer needs at least 200 rows for holdout-based SQS/DPS. `--allow-no-holdout` is available only for service smoke testing; it sets `holdout=0`, so quality/privacy scores are unavailable and the result must not be treated as production-evaluated.

## NeMo Data Designer tool-calling generation

This optional command uses the official structured-column flow and generates 500+ examples by default:

```bash
uv sync --extra platform
export NMP_BASE_URL=https://your-nmp.example
export NMP_MODEL_PROVIDER=default/nvidia-build
export NMP_MODEL_ID=nvidia/nemotron-3-nano-30b-a3b
uv run humana-sdg data-designer-tools --num-records 500
```

The command runs a four-record preview first, requires exactly one tool call, blocks real-person/member data in prompts and writes OpenAI-format JSONL.

## NeMo embedding customization

Generate page-grounded triplets with `humana-sdg generate`, create a NeMo Platform secret named `hf-token` (or set `NMP_HF_SECRET_NAME`), and submit:

```bash
uv sync --extra platform
export NMP_BASE_URL=https://your-nmp.example
export NMP_HF_SECRET_NAME=hf-token
uv run humana-sdg customize-embeddings \
  --dataset data/synthetic/embedding_triplets
```

The implementation follows the supplied NVIDIA tutorial:

- upload `training.jsonl` and `validation.jsonl` as a FileSet;
- register `nvidia/llama-nemotron-embed-1b-v2`;
- submit SFT with batch size 128, learning rate `5e-6`, max sequence length 512 and one GPU.

Secrets must already exist in the NeMo Platform secret store. The code never logs or commits their values.

## NeMo Evaluator Skills

The supplied Skills tutorial evaluates general model capability, not document grounding. For the tool-calling portion, BFCL v4 is relevant:

```bash
export NVIDIA_API_KEY=...
export NEMO_MODEL_URL=https://integrate.api.nvidia.com/v1/chat/completions
export NEMO_MODEL_ID=nvidia/nemotron-3-super-120b-a12b
MAX_PROBLEMS=20 bash scripts/run_nemo_evaluator_skills.sh
```

Use the built-in citation/privacy evaluator for this corpus and NeMo Evaluator BFCL v4 for deployed tool-calling model behavior. Do not substitute MMLU/GPQA scores for domain grounding evidence.

## Grounding PDFs

The canonical machine-readable list is [`corpus/manifest.json`](corpus/manifest.json). All 22 URLs were probed on 2026-07-21 and returned PDF content from official hosts.

### Humana

1. [Humana Provider Manual (Delegated Providers), 2026](https://assets.humana.com/is/content/humana/FINAL_773902ALL0725-A_2026_ProviderManual-Delegated_formattedpdf)
2. [Humana Provider Manual (Nondelegated Providers), 2025](https://assets.humana.com/is/content/humana/FINAL589003ALL1024_2025_ProviderManualNonDelegatedpdf)
3. [Humana Healthy Horizons in Oklahoma Provider Manual, 2025](https://assets.humana.com/is/content/humana/2025%20OK%20Provider%20Manual%206-11-2025pdf)
4. [Humana Healthy Horizons in South Carolina Provider Manual, 2025](https://assets.humana.com/is/content/humana/2025_SC_Provider_Manualpdf)
5. [Humana Healthy Horizons in Kentucky Provider Manual, 2025](https://assets.humana.com/is/content/humana/2025_KY_Provider_Manualpdf)
6. [Humana Healthy Horizons in Ohio Provider Manual, 2025](https://assets.humana.com/is/content/humana/2025_OH_Provider_Manualpdf)
7. [Humana HMO D-SNP Provider Billing Guide](https://assets.humana.com/is/content/humana/HMO%20D-SNP%20Provider%20Billing%20Guidepdf)
8. [Humana Dental Office Handbook, 2026](https://assets.humana.com/is/content/humana/Dental%20Office%20Handbook%202026pdf)
9. [Humana Medicare Prior Authorization and Notification List, July 2026](https://assets.humana.com/is/content/humana/July%202026%20Medicare%20Prior%20Authorization%20Listpdf)
10. [Humana Medicare Part B Step Therapy Preferred Drug List, 2026](https://assets.humana.com/is/content/humana/2026%20Part%20B%20step%20therapy%20preferred%20drug%20listpdf)
11. [Humana Prescriber Quick Reference Guide](https://assets.humana.com/is/content/humana/Prescriber%20Quick%20Reference%20Guidepdf)
12. [Humana Plan H5619-152-000 Evidence of Coverage, 2026](https://assets.humana.com/is/content/humana/H5619152000EOC26pdf)
13. [Humana Impact Report, 2025](https://assets.humana.com/is/content/humana/2025_Humana_Impact_Reportpdf)
14. [Humana Compliance Policy](https://assets.humana.com/is/content/humana/Compliance%20Policypdf)
15. [Humana Privacy Policy](https://assets.humana.com/is/content/humana/Humana%20Privacy%20Policypdf-2)
16. [Humana Supplier Code of Conduct, 2026](https://assets.humana.com/is/content/humana/Humana_Supplier_Code_of_Conduct_2026pdf)

### Medicare / CMS

17. [Medicare & You 2026](https://www.medicare.gov/publications/10050-medicare-and-you.pdf)
18. [Medicare Claims Processing Manual, Chapter 1: General Billing Requirements](https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/clm104c01.pdf)
19. [Medicare Benefit Policy Manual, Chapter 1: Inpatient Hospital Services](https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c01.pdf)
20. [Medicare Managed Care Manual, Chapter 13: Appeals and Grievances](https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/mc86c13.pdf)
21. [Medicare Parts A & B Appeals Process (MLN006562)](https://www.cms.gov/files/document/mln006562-medicare-parts-a-b-appeals-process.pdf)
22. [Medicare Managed Care Appeals Flow Chart](https://www.cms.gov/files/document/medicare-managed-care-appeals-flow-chart.pdf)

## NVIDIA references and compatibility

Implementation was checked against these supplied pages:

- [NeMo Curator Welcome](https://docs.nvidia.com/nemo/curator/latest/home/welcome)
- [Tool-Calling Fine-Tuning with Synthetic Data](https://docs.nvidia.com/nemo/microservices/latest/example-applications/tool-calling.html)
- [Embedding Model Customization](https://docs.nvidia.com/nemo/microservices/latest/customizer/tutorials/embedding-customization-job.html)
- [Safe Synthesizer 101, version 26.3.0](https://docs.nvidia.com/nemo/microservices/26.3.0/safe-synthesizer/tutorials/safe-synthesizer-101.html)
- [NeMo Evaluator Skills Integration](https://docs.nvidia.com/nemo/evaluator/tutorials/skills-integration)
- [Safe Data Evaluation](https://docs.nvidia.com/nemo-platform/documentation/synthesize-safe-data/about/evaluation)

The references mix `latest`, a fixed `26.3.0` tutorial and separately released Python packages. This project pins NeMo Curator `1.2.0`; NeMo Platform calls remain isolated behind the `platform` optional dependency. Verify SDK/server compatibility in your deployment before launching costly jobs.

## Development

```bash
uv sync --extra dev
uv run python -m pytest -q
uv run ruff check src tests
```

The test suite covers manifest allowlisting, PDF validation, page-preserving extraction, deterministic deduplication, grounded generation, fail-closed evaluation, Safe Synthesizer frame preparation, tool-call schema integrity and embedding triplets.

## Repository layout

```text
corpus/manifest.json                 verified link-only source corpus
src/humana_sdg/download.py           fail-closed PDF downloader
src/humana_sdg/extract.py            page extraction and chunking
src/humana_sdg/curate.py             NeMo Curator adapter + exact dedup
src/humana_sdg/generate.py           grounded deterministic generator
src/humana_sdg/support.py            multi-turn support transcripts + quality gates
src/humana_sdg/tool_data.py          governed OpenAI tool-call records
src/humana_sdg/embedding.py          Customizer triplet preparation
src/humana_sdg/evaluate.py           grounding/privacy gates
src/humana_sdg/safe_synth.py         Safe Synthesizer adapter
src/humana_sdg/nemo_platform_jobs.py Data Designer and Customizer adapters
cluster/run_humana_pipeline.sbatch   verified GPU/Slurm entrypoint
scripts/run_offline_smoke.sh         local end-to-end smoke
scripts/run_nemo_evaluator_skills.sh BFCL v4 Evaluator entrypoint
```

## Customer-support SLM transcript dataset

The production pipeline generates 1,200 deterministic, citation-preserving conversations (9,600 alternating turns) in `synthetic/support_conversations.jsonl`, plus `synthetic/support_sft_openai.jsonl` for SLM fine-tuning. The 12 balanced use cases are appeals/grievances, care management, claims/billing, compliance reporting, dental benefits, eligibility/enrollment, Medicaid member support, pharmacy benefits, plan benefits, prior authorization, privacy preferences, and provider directory.

Run the full NeMo Curator pipeline and transcript gates:

```bash
uv run humana-sdg all \
  --engine nemo \
  --records-per-chunk 2 \
  --support-conversations 1200 \
  --support-seed 20260722 \
  --workspace outputs/customer-support-full
```

Evaluate an existing transcript file:

```bash
uv run humana-sdg evaluate-support \
  --conversations outputs/customer-support-full/synthetic/support_conversations.jsonl \
  --curated outputs/customer-support-full/curated/chunks.jsonl \
  --output outputs/customer-support-full/support_evaluation.json
```

### Authentic Safe Synthesizer SQS and DPS

`synthetic_data_quality_score` (SQS) and `data_privacy_score` (DPS) are service-produced Safe Synthesizer metrics. This repository records those fields only from the official Safe Synthesizer job summary; it does not fabricate or relabel local heuristic scores. Configure an actual NeMo Platform deployment, then run:

```bash
export NMP_BASE_URL='https://<your-nemo-platform>'
export NMP_WORKSPACE='<workspace>'
export NMP_ACCESS_TOKEN='<runtime-secret>'
export NMP_MODEL_PROVIDER='<configured-provider>'
CONVERSATIONS=outputs/customer-support-full/synthetic/support_conversations.jsonl \
  bash scripts/run_nemo_safe_synth_conversations.sh
```

The adapter writes the generated dataset, evaluation report, raw report bundle, and `safe_synth_summary.json` containing authentic SQS/DPS values. NeMo Evaluator is a separate model-evaluation service: use it after a customer-support SLM endpoint has been trained and deployed. Dataset gates in `support_evaluation.json` are project quality evidence and are never presented as NeMo Evaluator service output.
