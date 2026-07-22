from __future__ import annotations

import json
from pathlib import Path

from humana_sdg.curate import write_chunks_jsonl
from humana_sdg.models import GroundingChunk
from humana_sdg.safe_synth import SUPPORT_SAFE_SYNTH_COLUMNS, conversations_to_safe_synth_frame
from humana_sdg.support import (
    SupportEvaluationThresholds,
    evaluate_support_conversations,
    generate_support_conversations,
    read_support_conversations_jsonl,
    write_support_datasets,
)
from humana_sdg.workflow import evaluate_support_dataset, generate_all_datasets

_CATEGORIES = (
    "provider_operations", "medicaid_provider_manual", "claims_billing",
    "dental_provider_operations", "prior_authorization", "pharmacy",
    "evidence_of_coverage", "privacy", "medicare_reference",
    "coverage_policy", "appeals_grievances",
)


def _chunks() -> list[GroundingChunk]:
    chunks = []
    for index, category in enumerate(_CATEGORIES, start=1):
        chunks.append(GroundingChunk(
            chunk_id=f"chunk-{index:02d}", source_id=f"official-source-{index:02d}",
            source_title=f"Official {category.replace('_', ' ').title()} Guide",
            source_url=f"https://assets.humana.com/is/content/humana/source-{index:02d}pdf",
            source_sha256=f"{index:064x}", publisher="Humana" if index < 10 else "CMS",
            year=2026, category=category, page=index,
            text=(f"The official {category.replace('_', ' ')} guide explains documented options and "
                  "requires the caller to verify current plan-specific requirements before action."),
            citation_label=f"Official guide, p. {index}",
        ))
    return chunks


def test_generates_1200_deterministic_diverse_grounded_conversations() -> None:
    first = generate_support_conversations(_chunks(), count=1200, seed=20260722)
    second = generate_support_conversations(_chunks(), count=1200, seed=20260722)
    assert first == second
    assert len(first) == 1200
    assert len({item.conversation_id for item in first}) == 1200
    assert len({item.use_case for item in first}) >= 10
    assert {item.channel for item in first} == {"chat", "phone-transcript", "secure-message"}
    assert {item.customer_persona for item in first} >= {
        "member", "caregiver", "provider-office", "prospective-member",
    }
    for conversation in first:
        assert 6 <= len(conversation.turns) <= 10
        assert conversation.turns[0].role == "customer"
        assert conversation.turns[-1].role == "agent"
        assert all(left.role != right.role for left, right in zip(
            conversation.turns, conversation.turns[1:], strict=False
        ))
        assert conversation.citations[0].quote in "\n".join(
            turn.content for turn in conversation.turns if turn.role == "agent"
        )
        assert conversation.is_synthetic is True
        assert conversation.requires_human_verification is True


def test_support_evaluator_scores_grounding_safety_privacy_and_diversity() -> None:
    chunks = _chunks()
    conversations = generate_support_conversations(chunks, count=120, seed=17)
    report = evaluate_support_conversations(
        conversations, chunks, SupportEvaluationThresholds(min_use_cases=10),
    )
    assert report.passed
    assert report.conversation_count == 120
    assert report.citation_valid_rate == 1.0
    assert report.grounded_quote_rate == 1.0
    assert report.turn_structure_rate == 1.0
    assert report.safety_compliance_rate == 1.0
    assert report.pii_findings == 0
    assert report.use_case_count >= 10

    unsafe = conversations[0].model_copy(deep=True)
    unsafe.turns[-1].content = (
        "You are guaranteed approval and coverage. Record the synthetic member as 123-45-6789."
    )
    failed = evaluate_support_conversations(
        [unsafe, *conversations[1:]], chunks, SupportEvaluationThresholds(min_use_cases=10),
    )
    assert not failed.passed
    assert failed.pii_findings == 1
    assert failed.safety_violation_count >= 1


def test_support_exports_full_and_openai_sft_jsonl(tmp_path: Path) -> None:
    conversations = generate_support_conversations(_chunks(), count=12, seed=9)
    paths = write_support_datasets(conversations, tmp_path)
    assert set(paths) == {"conversations", "openai_sft"}
    assert read_support_conversations_jsonl(paths["conversations"]) == conversations
    sft_rows = [json.loads(line) for line in paths["openai_sft"].read_text(encoding="utf-8").splitlines()]
    assert len(sft_rows) == 12
    assert all(row["messages"][0]["role"] == "user" for row in sft_rows)
    assert all(row["messages"][-1]["role"] == "assistant" for row in sft_rows)
    assert all(row["metadata"]["is_synthetic"] is True for row in sft_rows)
    assert all(row["metadata"]["citations"] for row in sft_rows)


def test_support_safe_synth_frame_uses_governed_transcript_columns() -> None:
    conversations = generate_support_conversations(_chunks(), count=24, seed=5)
    frame = conversations_to_safe_synth_frame(conversations)
    assert list(frame.columns) == SUPPORT_SAFE_SYNTH_COLUMNS
    assert len(frame) == 24
    assert frame["conversation_id"].is_unique
    assert frame["transcript"].str.contains("CUSTOMER:").all()
    assert frame["transcript"].str.contains("AGENT:").all()
    assert frame["is_synthetic"].all()
    assert frame["requires_human_verification"].all()



def test_full_dataset_workflow_generates_and_evaluates_1200_support_transcripts(tmp_path: Path) -> None:
    curated = tmp_path / "curated.jsonl"
    write_chunks_jsonl(_chunks(), curated)

    paths = generate_all_datasets(
        curated,
        tmp_path / "synthetic",
        records_per_chunk=1,
        tool_records=5,
        support_conversations=1200,
        support_seed=20260722,
    )

    assert {"support_conversations", "support_openai_sft"} <= set(paths)
    assert len(read_support_conversations_jsonl(paths["support_conversations"])) == 1200
    report = evaluate_support_dataset(
        paths["support_conversations"],
        curated,
        tmp_path / "support_evaluation.json",
    )
    assert report.passed
    assert report.conversation_count == 1200
    assert (tmp_path / "support_evaluation.json").exists()

def test_contextual_official_quote_is_not_misclassified_as_an_agent_guarantee() -> None:
    chunk = _chunks()[0].model_copy(
        update={
            "text": (
                "The official guide states: You are covered for emergency care world-wide. "
                "Verify current plan terms."
            )
        }
    )
    conversations = generate_support_conversations([chunk], count=1, seed=33)
    report = evaluate_support_conversations(
        conversations,
        [chunk],
        SupportEvaluationThresholds(min_use_cases=1),
    )
    assert report.safety_compliance_rate == 1.0
    assert report.passed




def test_grounded_negative_guarantee_language_is_not_a_safety_violation() -> None:
    chunk = _chunks()[0].model_copy(
        update={
            "category": "coverage_policy",
            "text": (
                "A member list does not prove eligibility for benefits or guarantee coverage. "
                "Verify current plan requirements through official methods."
            ),
        }
    )
    conversations = generate_support_conversations([chunk], count=12, seed=3)

    report = evaluate_support_conversations(
        conversations,
        [chunk],
        SupportEvaluationThresholds(min_use_cases=2),
    )

    assert report.passed
    assert report.safety_violation_count == 0
