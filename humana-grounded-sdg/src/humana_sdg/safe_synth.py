from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from humana_sdg.models import SyntheticRecord

SAFE_SYNTH_COLUMNS = [
    "record_id",
    "task_type",
    "question",
    "answer",
    "citation_source_ids",
    "citation_pages",
    "is_synthetic",
    "disclaimer",
]


class SafeSynthSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = "http://localhost:8080"
    workspace: str = "default"
    access_token: str | None = None
    project_name: str = "humana-grounded-sdg"
    provider_name: str = "system/nvidia-build"
    hf_secret_name: str | None = None
    allow_no_holdout: bool = False
    minimum_rows_for_evaluation: int = Field(default=200, ge=200)


class SafeSynthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_name: str
    synthetic_csv: Path
    evaluation_report: Path
    summary_json: Path
    synthetic_quality_score: float | None
    data_privacy_score: float | None


def records_to_safe_synth_frame(records: Iterable[SyntheticRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "record_id": record.record_id,
                "task_type": record.task_type,
                "question": record.question,
                "answer": record.answer,
                "citation_source_ids": "|".join(citation.source_id for citation in record.citations),
                "citation_pages": "|".join(str(citation.page) for citation in record.citations),
                "is_synthetic": record.is_synthetic,
                "disclaimer": record.disclaimer,
            }
        )
    return pd.DataFrame(rows, columns=SAFE_SYNTH_COLUMNS)


def run_safe_synthesis(
    frame: pd.DataFrame,
    output_directory: Path,
    settings: SafeSynthSettings,
) -> SafeSynthResult:
    """Run the official NeMo Platform Safe Synthesizer SDK workflow."""
    try:
        from nemo_platform import ConflictError, NeMoPlatform

        try:
            from nemo_safe_synthesizer_plugin.sdk.job_builder import SafeSynthesizerJobBuilder
        except ImportError:
            from nemo_platform.beta.safe_synthesizer.job_builder import SafeSynthesizerJobBuilder
    except ImportError as exc:
        raise RuntimeError("Install NeMo Platform: uv sync --extra platform") from exc

    if len(frame) < settings.minimum_rows_for_evaluation and not settings.allow_no_holdout:
        raise ValueError(
            f"Safe Synthesizer evaluation requires at least {settings.minimum_rows_for_evaluation} rows; "
            "generate more rows or explicitly set allow_no_holdout=True for a smoke run without SQS/DPS."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    client = NeMoPlatform(
        base_url=settings.base_url,
        workspace=settings.workspace,
        access_token=settings.access_token,
    )
    client.safe_synthesizer.jobs.list(workspace=settings.workspace)
    try:
        client.projects.create(workspace=settings.workspace, name=settings.project_name)
    except ConflictError:
        pass

    builder = (
        SafeSynthesizerJobBuilder(client, workspace=settings.workspace)
        .with_data_source(frame)
        .with_classify_model_provider(settings.provider_name)
        .with_replace_pii()
        .synthesize()
    )
    if len(frame) < settings.minimum_rows_for_evaluation:
        builder = builder.with_data(holdout=0)
    if settings.hf_secret_name:
        builder = builder.with_hf_token_secret(settings.hf_secret_name)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    job = builder.create_job(name=f"humana-grounded-{timestamp}", project=settings.project_name)
    job.wait_for_completion()
    synthetic_frame = job.fetch_data()
    summary = job.fetch_summary()

    synthetic_csv = output_directory / "safe_synthetic.csv"
    evaluation_report = output_directory / "safe_synth_evaluation.html"
    summary_json = output_directory / "safe_synth_summary.json"
    synthetic_frame.to_csv(synthetic_csv, index=False)
    job.save_report(str(evaluation_report))
    payload = {
        "job_name": job.job_name,
        "synthetic_data_quality_score": getattr(summary, "synthetic_data_quality_score", None),
        "data_privacy_score": getattr(summary, "data_privacy_score", None),
        "num_valid_records": getattr(summary, "num_valid_records", None),
        "num_prompts": getattr(summary, "num_prompts", None),
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return SafeSynthResult(
        job_name=job.job_name,
        synthetic_csv=synthetic_csv,
        evaluation_report=evaluation_report,
        summary_json=summary_json,
        synthetic_quality_score=payload["synthetic_data_quality_score"],
        data_privacy_score=payload["data_privacy_score"],
    )
