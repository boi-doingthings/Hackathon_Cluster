from __future__ import annotations

import json
from pathlib import Path

from humana_sdg.curate import (
    CuratorSettings,
    curate_chunks_python,
    read_chunks_jsonl,
    run_nemo_curator,
    write_chunks_jsonl,
)
from humana_sdg.download import DownloadReceipt, download_source
from humana_sdg.embedding import build_embedding_triplets, split_and_write_triplets
from humana_sdg.evaluate import EvaluationReport, EvaluationThresholds, evaluate_records
from humana_sdg.extract import extract_pdf
from humana_sdg.generate import generate_deterministic_records, write_records_jsonl
from humana_sdg.manifest import load_manifest
from humana_sdg.models import SyntheticRecord
from humana_sdg.support import (
    SupportEvaluationReport,
    SupportEvaluationThresholds,
    evaluate_support_conversations,
    generate_support_conversations,
    read_support_conversations_jsonl,
    write_support_datasets,
)
from humana_sdg.tool_data import generate_tool_call_records, write_tool_call_jsonl


def download_manifest_sources(
    manifest_path: Path,
    raw_directory: Path,
    *,
    limit: int | None = None,
    source_ids: set[str] | None = None,
) -> Path:
    manifest = load_manifest(manifest_path)
    selected = [source for source in manifest.sources if not source_ids or source.id in source_ids]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No sources selected")

    receipts = [download_source(source, raw_directory) for source in selected]
    receipt_path = raw_directory / "download_receipts.json"
    receipt_path.write_text(
        json.dumps([receipt.model_dump(mode="json") for receipt in receipts], indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt_path


def extract_downloaded_pdfs(
    manifest_path: Path,
    receipt_path: Path,
    output_jsonl: Path,
    *,
    max_words: int = 220,
    overlap_words: int = 30,
) -> Path:
    manifest = load_manifest(manifest_path)
    sources = {source.id: source for source in manifest.sources}
    receipts = _load_receipts(receipt_path)
    chunks = []
    for receipt in receipts:
        source = sources.get(receipt.source_id)
        if source is None:
            raise ValueError(f"Receipt source not found in manifest: {receipt.source_id}")
        chunks.extend(
            extract_pdf(
                source,
                receipt.path,
                pdf_sha256=receipt.sha256,
                max_words=max_words,
                overlap_words=overlap_words,
            )
        )
    write_chunks_jsonl(chunks, output_jsonl)
    return output_jsonl


def curate_extracted_chunks(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    engine: str,
    settings: CuratorSettings,
) -> Path:
    if engine == "nemo":
        curator_directory = output_jsonl.parent / "nemo_curator_parts"
        run = run_nemo_curator(input_jsonl, curator_directory, settings)
        chunks = read_chunks_jsonl(run.output_files)
    elif engine == "python":
        chunks = read_chunks_jsonl(input_jsonl)
    else:
        raise ValueError("engine must be 'nemo' or 'python'")

    curated = curate_chunks_python(chunks, settings)
    if not curated:
        raise RuntimeError("No chunks remained after curation")
    write_chunks_jsonl(curated, output_jsonl)
    return output_jsonl


def generate_all_datasets(
    curated_jsonl: Path,
    output_directory: Path,
    *,
    records_per_chunk: int = 2,
    tool_records: int = 200,
    support_conversations: int = 1200,
    support_seed: int = 20260722,
) -> dict[str, Path]:
    chunks = read_chunks_jsonl(curated_jsonl)
    output_directory.mkdir(parents=True, exist_ok=True)

    records_path = output_directory / "grounded_synthetic.jsonl"
    records = generate_deterministic_records(chunks, records_per_chunk=records_per_chunk)
    write_records_jsonl(records, records_path)

    tool_path = output_directory / "tool_calling_openai.jsonl"
    write_tool_call_jsonl(generate_tool_call_records(tool_records), tool_path)

    conversations = generate_support_conversations(
        chunks, count=support_conversations, seed=support_seed
    )
    support_paths = write_support_datasets(conversations, output_directory)

    embedding_directory = output_directory / "embedding_triplets"
    triplets = build_embedding_triplets(chunks)
    training_path, validation_path = split_and_write_triplets(triplets, embedding_directory)
    return {
        "grounded_records": records_path,
        "support_conversations": support_paths["conversations"],
        "support_openai_sft": support_paths["openai_sft"],
        "tool_calling": tool_path,
        "embedding_training": training_path,
        "embedding_validation": validation_path,
    }


def evaluate_support_dataset(
    conversations_jsonl: Path,
    curated_jsonl: Path,
    report_path: Path,
    *,
    thresholds: SupportEvaluationThresholds | None = None,
) -> SupportEvaluationReport:
    conversations = read_support_conversations_jsonl(conversations_jsonl)
    chunks = read_chunks_jsonl(curated_jsonl)
    report = evaluate_support_conversations(
        conversations, chunks, thresholds or SupportEvaluationThresholds()
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_dataset(
    records_jsonl: Path,
    curated_jsonl: Path,
    report_path: Path,
    *,
    thresholds: EvaluationThresholds | None = None,
) -> EvaluationReport:
    records = read_records_jsonl(records_jsonl)
    chunks = read_chunks_jsonl(curated_jsonl)
    report = evaluate_records(records, chunks, thresholds or EvaluationThresholds())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def read_records_jsonl(path: Path) -> list[SyntheticRecord]:
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(SyntheticRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid synthetic record at {path}:{line_number}: {exc}") from exc
    return records


def _load_receipts(path: Path) -> list[DownloadReceipt]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DownloadReceipt.model_validate(item) for item in payload]
