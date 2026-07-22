from __future__ import annotations

from typer.testing import CliRunner

from humana_sdg.cli import app
from humana_sdg.curate import write_chunks_jsonl
from humana_sdg.models import GroundingChunk
from humana_sdg.support import read_support_conversations_jsonl


def test_generate_cli_writes_requested_conversation_count(tmp_path) -> None:
    categories = (
        "evidence_of_coverage",
        "claims_billing",
        "compliance",
        "appeals_grievances",
        "medicare_reference",
        "pharmacy",
        "provider_operations",
        "corporate_impact",
    )
    chunks = [
        GroundingChunk(
            chunk_id=f"cli-{index}",
            source_id=f"source-{index}",
            source_title=f"Official Source {index}",
            source_url=f"https://humana.com/source-{index}.pdf",
            source_sha256=f"{index:x}" * 64,
            publisher="Humana",
            year=2026,
            category=category,
            page=index,
            text=f"Official {category} guidance says to verify the current plan document.",
            citation_label=f"Official Source {index}, p. {index}",
        )
        for index, category in enumerate(categories, start=1)
    ]
    curated = tmp_path / "curated.jsonl"
    output = tmp_path / "synthetic"
    write_chunks_jsonl(chunks, curated)

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--curated",
            str(curated),
            "--output",
            str(output),
            "--records-per-chunk",
            "1",
            "--tool-records",
            "8",
            "--conversation-records",
            "1001",
        ],
    )

    assert result.exit_code == 0, result.output
    transcripts = output / "support_conversations.jsonl"
    assert len(read_support_conversations_jsonl(transcripts)) == 1001

def test_safe_synthesize_conversations_cli_is_available() -> None:
    result = CliRunner().invoke(app, ["safe-synthesize-conversations", "--help"])

    assert result.exit_code == 0, result.output
    assert "--conversations" in result.output

