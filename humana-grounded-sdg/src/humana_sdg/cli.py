from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from humana_sdg.curate import CuratorSettings
from humana_sdg.nemo_platform_jobs import (
    DataDesignerSettings,
    EmbeddingCustomizationSettings,
    PlatformConnection,
    run_data_designer_tool_calling,
    submit_embedding_customization,
)
from humana_sdg.safe_synth import SafeSynthSettings, records_to_safe_synth_frame, run_safe_synthesis
from humana_sdg.workflow import (
    curate_extracted_chunks,
    download_manifest_sources,
    evaluate_dataset,
    extract_downloaded_pdfs,
    generate_all_datasets,
    read_records_jsonl,
)

app = typer.Typer(no_args_is_help=True, help="NeMo-first Humana grounded synthetic-data pipeline.")
DEFAULT_MANIFEST = Path("corpus/manifest.json")


@app.command()
def download(
    manifest: Path = typer.Option(DEFAULT_MANIFEST, exists=True),
    output: Path = typer.Option(Path("data/raw")),
    limit: int | None = typer.Option(None, min=1),
) -> None:
    receipt_path = download_manifest_sources(manifest, output, limit=limit)
    typer.echo(receipt_path)


@app.command()
def extract(
    manifest: Path = typer.Option(DEFAULT_MANIFEST, exists=True),
    receipts: Path = typer.Option(Path("data/raw/download_receipts.json"), exists=True),
    output: Path = typer.Option(Path("data/interim/extracted_chunks.jsonl")),
    max_words: int = typer.Option(220, min=20),
    overlap_words: int = typer.Option(30, min=0),
) -> None:
    typer.echo(
        extract_downloaded_pdfs(
            manifest,
            receipts,
            output,
            max_words=max_words,
            overlap_words=overlap_words,
        )
    )


@app.command()
def curate(
    input_path: Path = typer.Option(Path("data/interim/extracted_chunks.jsonl"), exists=True),
    output: Path = typer.Option(Path("data/curated/chunks.jsonl")),
    engine: str = typer.Option("nemo", help="nemo (required production path) or python (offline smoke only)"),
    min_words: int = typer.Option(40, min=1),
    max_words: int = typer.Option(350, min=1),
) -> None:
    path = curate_extracted_chunks(
        input_path,
        output,
        engine=engine,
        settings=CuratorSettings(min_words=min_words, max_words=max_words),
    )
    typer.echo(path)


@app.command()
def generate(
    curated: Path = typer.Option(Path("data/curated/chunks.jsonl"), exists=True),
    output: Path = typer.Option(Path("data/synthetic")),
    records_per_chunk: int = typer.Option(2, min=1, max=3),
    tool_records: int = typer.Option(200, min=1),
) -> None:
    paths = generate_all_datasets(
        curated,
        output,
        records_per_chunk=records_per_chunk,
        tool_records=tool_records,
    )
    typer.echo(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


@app.command()
def evaluate(
    records: Path = typer.Option(Path("data/synthetic/grounded_synthetic.jsonl"), exists=True),
    curated: Path = typer.Option(Path("data/curated/chunks.jsonl"), exists=True),
    output: Path = typer.Option(Path("outputs/evaluation.json")),
) -> None:
    report = evaluate_dataset(records, curated, output)
    typer.echo(report.model_dump_json(indent=2))
    if not report.passed:
        raise typer.Exit(code=2)


def _platform_connection() -> PlatformConnection:
    return PlatformConnection(
        base_url=os.environ.get("NMP_BASE_URL", "http://localhost:8080"),
        workspace=os.environ.get("NMP_WORKSPACE", "default"),
        access_token=os.environ.get("NMP_ACCESS_TOKEN"),
    )


@app.command("data-designer-tools")
def data_designer_tools(
    output: Path = typer.Option(Path("data/synthetic/tool_calling_nemo.jsonl")),
    num_records: int = typer.Option(500, min=1),
) -> None:
    settings = DataDesignerSettings(
        provider=os.environ.get("NMP_MODEL_PROVIDER", "default/nvidia-build"),
        model=os.environ.get("NMP_MODEL_ID", "nvidia/nemotron-3-nano-30b-a3b"),
        num_records=num_records,
    )
    typer.echo(run_data_designer_tool_calling(_platform_connection(), settings, output))


@app.command("customize-embeddings")
def customize_embeddings(
    dataset: Path = typer.Option(Path("data/synthetic/embedding_triplets"), exists=True),
) -> None:
    settings = EmbeddingCustomizationSettings(hf_secret_name=os.environ.get("NMP_HF_SECRET_NAME", "hf-token"))
    typer.echo(
        json.dumps(
            submit_embedding_customization(_platform_connection(), settings, dataset),
            indent=2,
        )
    )


@app.command("safe-synthesize")
def safe_synthesize(
    records: Path = typer.Option(Path("data/synthetic/grounded_synthetic.jsonl"), exists=True),
    output: Path = typer.Option(Path("outputs/safe_synth")),
    allow_no_holdout: bool = typer.Option(False),
) -> None:
    settings = SafeSynthSettings(
        base_url=os.environ.get("NMP_BASE_URL", "http://localhost:8080"),
        workspace=os.environ.get("NMP_WORKSPACE", "default"),
        access_token=os.environ.get("NMP_ACCESS_TOKEN"),
        provider_name=os.environ.get("NMP_MODEL_PROVIDER", "system/nvidia-build"),
        hf_secret_name=os.environ.get("NMP_HF_SECRET_NAME"),
        allow_no_holdout=allow_no_holdout,
    )
    frame = records_to_safe_synth_frame(read_records_jsonl(records))
    typer.echo(run_safe_synthesis(frame, output, settings).model_dump_json(indent=2))


@app.command("all")
def run_all(
    workspace: Path = typer.Option(Path("outputs/run")),
    manifest: Path = typer.Option(DEFAULT_MANIFEST, exists=True),
    engine: str = typer.Option("nemo"),
    limit: int | None = typer.Option(None, min=1),
    records_per_chunk: int = typer.Option(2, min=1, max=3),
) -> None:
    raw = workspace / "raw"
    interim = workspace / "interim" / "extracted_chunks.jsonl"
    curated = workspace / "curated" / "chunks.jsonl"
    synthetic = workspace / "synthetic"
    receipt_path = download_manifest_sources(manifest, raw, limit=limit)
    extract_downloaded_pdfs(manifest, receipt_path, interim)
    curate_extracted_chunks(
        interim,
        curated,
        engine=engine,
        settings=CuratorSettings(),
    )
    paths = generate_all_datasets(curated, synthetic, records_per_chunk=records_per_chunk)
    report = evaluate_dataset(paths["grounded_records"], curated, workspace / "evaluation.json")
    typer.echo(
        json.dumps(
            {
                "workspace": str(workspace),
                "records": str(paths["grounded_records"]),
                "evaluation_passed": report.passed,
                "record_count": report.record_count,
            },
            indent=2,
        )
    )
    if not report.passed:
        raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
