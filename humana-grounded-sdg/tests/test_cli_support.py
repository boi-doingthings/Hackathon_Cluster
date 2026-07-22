from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from humana_sdg.cli import app
from humana_sdg.curate import write_chunks_jsonl
from humana_sdg.models import GroundingChunk


def _chunk(index: int, category: str) -> GroundingChunk:
    return GroundingChunk(
        chunk_id=f"cli-chunk-{index}",
        source_id=f"cli-source-{index}",
        source_title=f"Official {category} guide",
        source_url=f"https://assets.humana.com/is/content/humana/cli-{index}pdf",
        source_sha256=f"{index:064x}",
        publisher="Humana",
        year=2026,
        category=category,
        page=index,
        text=f"Official {category} guidance requires verification against current plan documents.",
        citation_label=f"Official guide, p. {index}",
    )


def test_cli_generates_and_evaluates_support_dataset(tmp_path: Path) -> None:
    categories = [
        "provider_operations", "medicaid_provider_manual", "claims_billing",
        "dental_provider_operations", "prior_authorization", "pharmacy",
        "evidence_of_coverage", "privacy", "medicare_reference",
        "coverage_policy", "appeals_grievances",
    ]
    curated = tmp_path / "chunks.jsonl"
    synthetic = tmp_path / "synthetic"
    report = tmp_path / "support_evaluation.json"
    write_chunks_jsonl([_chunk(index, category) for index, category in enumerate(categories, 1)], curated)
    runner = CliRunner()

    generated = runner.invoke(app, [
        "generate", "--curated", str(curated), "--output", str(synthetic),
        "--records-per-chunk", "1", "--tool-records", "2",
        "--support-conversations", "1000", "--support-seed", "42",
    ])
    assert generated.exit_code == 0, generated.output
    payload = json.loads(generated.output)
    support_path = Path(payload["support_conversations"])
    assert support_path.exists()
    assert sum(1 for line in support_path.read_text(encoding="utf-8").splitlines() if line) == 1000

    evaluated = runner.invoke(app, [
        "evaluate-support", "--conversations", str(support_path),
        "--curated", str(curated), "--output", str(report),
    ])
    assert evaluated.exit_code == 0, evaluated.output
    evaluation = json.loads(report.read_text(encoding="utf-8"))
    assert evaluation["passed"] is True
    assert evaluation["conversation_count"] == 1000
    assert evaluation["use_case_count"] >= 10
