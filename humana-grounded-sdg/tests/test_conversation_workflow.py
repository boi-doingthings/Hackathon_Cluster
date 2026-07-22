from __future__ import annotations

from humana_sdg.conversations import ConversationEvaluationThresholds, read_conversations_jsonl
from humana_sdg.curate import write_chunks_jsonl
from humana_sdg.models import GroundingChunk
from humana_sdg.workflow import evaluate_conversation_dataset, generate_conversation_dataset


def test_workflow_generates_and_evaluates_customer_support_transcripts(tmp_path) -> None:
    chunks = []
    categories = (
        "benefits_eligibility",
        "claims_billing",
        "compliance_fraud_waste_abuse",
        "appeals_grievances",
        "medicare_basics",
        "pharmacy",
        "provider_contracting",
        "quality_value_based_care",
    )
    for index, category in enumerate(categories, start=1):
        chunks.append(
            GroundingChunk(
                chunk_id=f"workflow-{index}",
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
        )
    curated = tmp_path / "curated.jsonl"
    transcripts = tmp_path / "customer_support_transcripts.jsonl"
    report_path = tmp_path / "conversation_evaluation.json"
    write_chunks_jsonl(chunks, curated)

    generated = generate_conversation_dataset(curated, transcripts, target_count=1001)
    report = evaluate_conversation_dataset(
        generated,
        curated,
        report_path,
        thresholds=ConversationEvaluationThresholds(min_conversations=1000, min_use_cases=8),
    )

    assert len(read_conversations_jsonl(generated)) == 1001
    assert report.passed
    assert report_path.exists()
