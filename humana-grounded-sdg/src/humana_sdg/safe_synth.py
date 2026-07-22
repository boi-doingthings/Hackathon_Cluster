from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from humana_sdg.models import SyntheticRecord
from humana_sdg.support import SupportConversation

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

SUPPORT_SAFE_SYNTH_COLUMNS = [
    "conversation_id",
    "use_case",
    "channel",
    "customer_persona",
    "customer_sentiment",
    "outcome",
    "transcript",
    "citation_source_ids",
    "citation_pages",
    "is_synthetic",
    "requires_human_verification",
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
    sqs: float | None = None
    dps: float | None = None
    official_scores_available: bool = False
    score_source: str = "nemo-safe-synthesizer"


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


def conversations_to_safe_synth_frame(
    conversations: Iterable[SupportConversation],
) -> pd.DataFrame:
    rows = []
    for conversation in conversations:
        transcript = "\n".join(
            f"{turn.role.upper()}: {turn.content}" for turn in conversation.turns
        )
        rows.append(
            {
                "conversation_id": conversation.conversation_id,
                "use_case": conversation.use_case,
                "channel": conversation.channel,
                "customer_persona": conversation.customer_persona,
                "customer_sentiment": conversation.customer_sentiment,
                "outcome": conversation.outcome,
                "transcript": transcript,
                "citation_source_ids": "|".join(
                    citation.source_id for citation in conversation.citations
                ),
                "citation_pages": "|".join(
                    str(citation.page) for citation in conversation.citations
                ),
                "is_synthetic": conversation.is_synthetic,
                "requires_human_verification": conversation.requires_human_verification,
                "disclaimer": conversation.disclaimer,
            }
        )
    return pd.DataFrame(rows, columns=SUPPORT_SAFE_SYNTH_COLUMNS)


def build_safe_synth_score_payload(summary: object, *, require_scores: bool) -> dict[str, object]:
    if hasattr(summary, "model_dump"):
        raw_summary = summary.model_dump(mode="json")
    elif isinstance(summary, dict):
        raw_summary = dict(summary)
    else:
        raw_summary = {
            name: getattr(summary, name, None)
            for name in (
                "synthetic_data_quality_score",
                "data_privacy_score",
                "num_valid_records",
                "num_invalid_records",
                "num_prompts",
                "valid_record_fraction",
            )
        }
    sqs = raw_summary.get(
        "synthetic_data_quality_score",
        getattr(summary, "synthetic_data_quality_score", None),
    )
    dps = raw_summary.get(
        "data_privacy_score",
        getattr(summary, "data_privacy_score", None),
    )
    available = sqs is not None and dps is not None
    payload: dict[str, object] = {
        "score_source": "nemo-safe-synthesizer",
        "official_scores_available": available,
        "sqs": sqs,
        "dps": dps,
        "synthetic_data_quality_score": sqs,
        "data_privacy_score": dps,
        "sqs_scale": "0-10",
        "dps_scale": "raw NeMo Safe Synthesizer score",
        "num_valid_records": raw_summary.get("num_valid_records"),
        "num_prompts": raw_summary.get("num_prompts"),
        "raw_summary": raw_summary,
    }
    if require_scores and not available:
        raise RuntimeError(
            "NeMo Safe Synthesizer completed but did not return required SQS/DPS; "
            "do not treat this run as production-evaluated."
        )
    return payload


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
    require_scores = len(frame) >= settings.minimum_rows_for_evaluation
    payload = build_safe_synth_score_payload(summary, require_scores=False)
    payload["job_name"] = job.job_name
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if require_scores and not payload["official_scores_available"]:
        raise RuntimeError(
            "NeMo Safe Synthesizer completed but did not return required SQS/DPS; "
            f"inspect {summary_json}."
        )
    return SafeSynthResult(
        job_name=job.job_name,
        synthetic_csv=synthetic_csv,
        evaluation_report=evaluation_report,
        summary_json=summary_json,
        synthetic_quality_score=payload["synthetic_data_quality_score"],
        data_privacy_score=payload["data_privacy_score"],
        sqs=payload["sqs"],
        dps=payload["dps"],
        official_scores_available=bool(payload["official_scores_available"]),
        score_source=str(payload["score_source"]),
    )
