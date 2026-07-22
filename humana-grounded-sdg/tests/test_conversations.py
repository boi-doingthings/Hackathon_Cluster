from __future__ import annotations

from copy import deepcopy

from humana_sdg.conversations import (
    ConversationEvaluationThresholds,
    evaluate_conversations,
    generate_support_conversations,
    read_conversations_jsonl,
    write_conversations_jsonl,
)
from humana_sdg.models import GroundingChunk

CATEGORIES = (
    "benefits_eligibility",
    "claims_billing",
    "compliance_fraud_waste_abuse",
    "appeals_grievances",
    "medicare_basics",
    "pharmacy",
    "provider_contracting",
    "quality_value_based_care",
    "corporate_impact",
    "medicaid_provider_manual",
    "privacy",
)


def _chunks() -> list[GroundingChunk]:
    chunks = []
    for index, category in enumerate(CATEGORIES, start=1):
        chunks.append(
            GroundingChunk(
                chunk_id=f"chunk-{index}",
                source_id=f"official-source-{index}",
                source_title=f"Official Humana or CMS Guide {index}",
                source_url=f"https://www.humana.com/guide-{index}.pdf",
                source_sha256=f"{index:x}" * 64,
                publisher="Humana" if index % 2 else "CMS",
                year=2026,
                category=category,
                page=index,
                text=(
                    f"Official guidance for {category.replace('_', ' ')} requires checking the current "
                    "plan document and using the published support process before communicating an outcome."
                ),
                citation_label=f"Official Guide {index}, p. {index}",
            )
        )
    return chunks


def test_generator_creates_1000_plus_diverse_grounded_conversations() -> None:
    first = generate_support_conversations(_chunks(), target_count=1001)
    second = generate_support_conversations(_chunks(), target_count=1001)

    assert first == second
    assert len(first) == 1001
    assert len({item.conversation_id for item in first}) == 1001
    use_cases = {item.use_case for item in first}
    assert len(use_cases) >= 11
    assert {"community_corporate_impact", "medicaid_provider_support", "privacy_data_rights"} <= use_cases
    assert all(len(item.turns) == 6 for item in first)
    assert all([turn.role for turn in item.turns] == ["customer", "assistant"] * 3 for item in first)
    assert all([turn.turn_index for turn in item.turns] == list(range(1, 7)) for item in first)
    assert all(item.citations[0].quote in " ".join(turn.content for turn in item.turns) for item in first)
    assert all(item.synthetic_only and item.requires_human_verification for item in first)


def test_conversation_evaluation_passes_safe_grounded_dataset() -> None:
    conversations = generate_support_conversations(_chunks(), target_count=1001)

    report = evaluate_conversations(
        conversations,
        _chunks(),
        ConversationEvaluationThresholds(min_conversations=1000, min_use_cases=8),
    )

    assert report.passed
    assert report.conversation_count == 1001
    assert report.citation_valid_rate == 1.0
    assert report.grounded_quote_rate == 1.0
    assert report.role_alternation_rate == 1.0
    assert report.human_verification_rate == 1.0
    assert report.duplicate_rate == 0.0
    assert report.pii_findings == 0


def test_conversation_evaluation_rejects_pii_and_broken_alternation() -> None:
    conversations = generate_support_conversations(_chunks(), target_count=8)
    unsafe = deepcopy(conversations)
    unsafe[0].turns[0].content += " My SSN is 123-45-6789."
    unsafe[1].turns[1].role = "customer"

    report = evaluate_conversations(
        unsafe,
        _chunks(),
        ConversationEvaluationThresholds(min_conversations=8, min_use_cases=8),
    )

    assert not report.passed
    assert report.pii_findings == 1
    assert report.role_alternation_rate < 1.0


def test_conversation_jsonl_round_trip(tmp_path) -> None:
    expected = generate_support_conversations(_chunks(), target_count=16)
    output = tmp_path / "customer_support_transcripts.jsonl"

    assert write_conversations_jsonl(expected, output) == 16
    assert read_conversations_jsonl(output) == expected
