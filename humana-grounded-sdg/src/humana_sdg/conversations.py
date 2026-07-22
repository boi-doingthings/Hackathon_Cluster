from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from humana_sdg.evaluate import _count_pii
from humana_sdg.models import Citation, GroundingChunk

DISCLAIMER = (
    "Synthetic customer-support training conversation grounded in cited public guidance; "
    "not a coverage decision, claim outcome, medical advice, or a substitute for the current plan document."
)
USE_CASES = (
    "benefits_eligibility",
    "claims_billing",
    "compliance_fraud_waste_abuse",
    "appeals_grievances",
    "medicare_basics",
    "pharmacy_part_d",
    "provider_network",
    "quality_value_based_care",
    "community_corporate_impact",
    "medicaid_provider_support",
    "privacy_data_rights",
)
UseCase = Literal[
    "benefits_eligibility",
    "claims_billing",
    "compliance_fraud_waste_abuse",
    "appeals_grievances",
    "medicare_basics",
    "pharmacy_part_d",
    "provider_network",
    "quality_value_based_care",
    "community_corporate_impact",
    "medicaid_provider_support",
    "privacy_data_rights",
]
Channel = Literal["phone", "chat", "secure_message"]
Role = Literal["customer", "assistant"]

CATEGORY_TO_USE_CASE: dict[str, UseCase] = {
    "benefits_eligibility": "benefits_eligibility",
    "evidence_of_coverage": "benefits_eligibility",
    "coverage_policy": "benefits_eligibility",
    "claims_billing": "claims_billing",
    "compliance": "compliance_fraud_waste_abuse",
    "compliance_fraud_waste_abuse": "compliance_fraud_waste_abuse",
    "appeals_grievances": "appeals_grievances",
    "medicare_basics": "medicare_basics",
    "pharmacy": "pharmacy_part_d",
    "pharmacy_part_d": "pharmacy_part_d",
    "prior_authorization": "pharmacy_part_d",
    "provider_contracting": "provider_network",
    "provider_operations": "provider_network",
    "dental_provider_operations": "provider_network",
    "quality": "quality_value_based_care",
    "quality_value_based_care": "quality_value_based_care",
    "corporate_impact": "community_corporate_impact",
    "medicaid_provider_manual": "medicaid_provider_support",
    "privacy": "privacy_data_rights",
}
OPENINGS: dict[str, str] = {
    "benefits_eligibility": "I need help understanding where to verify a benefit or eligibility question.",
    "claims_billing": "I need help understanding the published process for a claim or billing question.",
    "compliance_fraud_waste_abuse": "I want to understand the official compliance or reporting guidance.",
    "appeals_grievances": "I need help finding the official appeals or grievance process.",
    "medicare_basics": "I have a general Medicare plan question and want current official guidance.",
    "pharmacy_part_d": "I need help locating current pharmacy or Part D guidance.",
    "provider_network": "I need help finding the official provider or network process.",
    "quality_value_based_care": "I want to understand the published quality or value-based care guidance.",
    "community_corporate_impact": "I need Humana's official community-impact guidance.",
    "medicaid_provider_support": "I need help finding the official Medicaid provider-support process.",
    "privacy_data_rights": "I need official privacy or data-rights guidance.",
}
AGE_BANDS = ("18-34", "35-49", "50-64", "65-74", "75+")
STATES = ("AZ", "FL", "KY", "OH", "TX")
PLAN_TYPES = ("Medicare Advantage", "Medicare Part D", "Commercial", "Dental")
CHANNELS: tuple[Channel, ...] = ("phone", "chat", "secure_message")


class CustomerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    age_band: str
    state: str
    plan_type: str
    channel: Channel


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=1)
    role: Role
    content: str = Field(min_length=1)


class SupportConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_reference: str
    use_case: UseCase
    customer_profile: CustomerProfile
    turns: list[ConversationTurn] = Field(min_length=6)
    citations: list[Citation] = Field(min_length=1)
    resolution: Literal["guidance_and_human_verification"] = "guidance_and_human_verification"
    synthetic_only: Literal[True] = True
    requires_human_verification: Literal[True] = True
    disclaimer: str


class ConversationEvaluationThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_conversations: int = Field(default=1000, ge=1)
    min_use_cases: int = Field(default=8, ge=1)
    min_citation_valid_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_grounded_quote_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_role_alternation_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_human_verification_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    max_duplicate_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_pii_findings: int = Field(default=0, ge=0)


class ConversationEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_count: int
    turn_count: int
    citation_count: int
    citation_valid_rate: float
    grounded_quote_rate: float
    role_alternation_rate: float
    human_verification_rate: float
    duplicate_rate: float
    pii_findings: int
    use_case_counts: dict[str, int]
    passed: bool
    failures: list[str]


def generate_support_conversations(
    chunks: Iterable[GroundingChunk],
    *,
    target_count: int = 1200,
) -> list[SupportConversation]:
    if target_count < 1:
        raise ValueError("target_count must be positive")
    grouped: dict[str, list[GroundingChunk]] = defaultdict(list)
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        grouped[_use_case_for_category(chunk.category)].append(chunk)
    active_use_cases = [use_case for use_case in USE_CASES if grouped[use_case]]
    if not active_use_cases:
        raise ValueError("At least one grounding chunk is required")

    conversations = []
    for ordinal in range(target_count):
        use_case = active_use_cases[ordinal % len(active_use_cases)]
        use_case_round = ordinal // len(active_use_cases)
        candidates = grouped[use_case]
        chunk = candidates[use_case_round % len(candidates)]
        conversations.append(_build_conversation(chunk, use_case, ordinal + 1))
    return conversations


def _use_case_for_category(category: str) -> UseCase:
    normalized = category.casefold().replace("-", "_").replace(" ", "_")
    if normalized in CATEGORY_TO_USE_CASE:
        return CATEGORY_TO_USE_CASE[normalized]
    if any(token in normalized for token in ("drug", "pharmacy", "part_d", "authorization")):
        return "pharmacy_part_d"
    if any(token in normalized for token in ("appeal", "grievance")):
        return "appeals_grievances"
    if any(token in normalized for token in ("claim", "billing", "payment")):
        return "claims_billing"
    if any(token in normalized for token in ("provider", "network", "dental")):
        return "provider_network"
    if any(token in normalized for token in ("quality", "value")):
        return "quality_value_based_care"
    if any(token in normalized for token in ("fraud", "waste", "abuse", "compliance")):
        return "compliance_fraud_waste_abuse"
    if "medicare" in normalized:
        return "medicare_basics"
    return "benefits_eligibility"


def _build_conversation(chunk: GroundingChunk, use_case: str, ordinal: int) -> SupportConversation:
    case_reference = f"SYNCASE{ordinal:06d}"
    quote = _grounded_excerpt(chunk.text)
    citation = Citation(
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        source_url=chunk.source_url,
        page=chunk.page,
        quote=quote,
    )
    profile = CustomerProfile(
        age_band=AGE_BANDS[(ordinal - 1) % len(AGE_BANDS)],
        state=STATES[(ordinal - 1) % len(STATES)],
        plan_type=PLAN_TYPES[(ordinal - 1) % len(PLAN_TYPES)],
        channel=CHANNELS[(ordinal - 1) % len(CHANNELS)],
    )
    turns = [
        ConversationTurn(
            turn_index=1,
            role="customer",
            content=f"Synthetic case {case_reference}: {OPENINGS[use_case]}",
        ),
        ConversationTurn(
            turn_index=2,
            role="assistant",
            content=(
                "I can explain the cited public guidance, but I cannot make a coverage, eligibility, "
                "claim, authorization, or medical decision."
            ),
        ),
        ConversationTurn(
            turn_index=3,
            role="customer",
            content=(
                f"For {case_reference}, please use only public guidance for a fictional "
                f"{profile.plan_type} scenario in {profile.state}."
            ),
        ),
        ConversationTurn(
            turn_index=4,
            role="assistant",
            content=f"Grounded guidance from {chunk.citation_label}: {quote}",
        ),
        ConversationTurn(
            turn_index=5,
            role="customer",
            content=f"What is the safest next step for this synthetic case {case_reference}?",
        ),
        ConversationTurn(
            turn_index=6,
            role="assistant",
            content=(
                "Verify the current member-specific plan document and official channel, avoid sending PHI "
                "in this training transcript, and escalate to an authorized Humana representative when a "
                "decision or case-specific interpretation is required."
            ),
        ),
    ]
    identity = hashlib.sha256(f"{chunk.chunk_id}\n{use_case}\n{ordinal}\n{quote}".encode()).hexdigest()
    return SupportConversation(
        conversation_id=identity,
        case_reference=case_reference,
        use_case=use_case,
        customer_profile=profile,
        turns=turns,
        citations=[citation],
        synthetic_only=True,
        requires_human_verification=True,
        disclaimer=DISCLAIMER,
    )


def evaluate_conversations(
    conversations: Iterable[SupportConversation],
    chunks: Iterable[GroundingChunk],
    thresholds: ConversationEvaluationThresholds | None = None,
) -> ConversationEvaluationReport:
    limits = thresholds or ConversationEvaluationThresholds()
    rows = list(conversations)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    citation_count = valid_citations = grounded_quotes = 0
    alternating = human_verified = pii_findings = 0
    fingerprints = []

    for conversation in rows:
        transcript = "\n".join(turn.content for turn in conversation.turns)
        pii_findings += _count_pii(transcript)
        expected_roles = [
            "customer" if index % 2 else "assistant" for index in range(1, len(conversation.turns) + 1)
        ]
        expected_indices = list(range(1, len(conversation.turns) + 1))
        if (
            len(conversation.turns) >= 6
            and [turn.role for turn in conversation.turns] == expected_roles
            and [turn.turn_index for turn in conversation.turns] == expected_indices
        ):
            alternating += 1
        if (
            conversation.requires_human_verification
            and conversation.synthetic_only
            and conversation.disclaimer
        ):
            human_verified += 1
        for citation in conversation.citations:
            citation_count += 1
            chunk = chunk_by_id.get(citation.chunk_id)
            if chunk is not None and (citation.source_id, citation.page, citation.source_url) == (
                chunk.source_id,
                chunk.page,
                chunk.source_url,
            ):
                valid_citations += 1
                if _normalize(citation.quote) in _normalize(chunk.text):
                    grounded_quotes += 1
        fingerprints.append(hashlib.sha256(_normalize(transcript).encode()).hexdigest())

    citation_valid_rate = _rate(valid_citations, citation_count)
    grounded_quote_rate = _rate(grounded_quotes, citation_count)
    role_alternation_rate = _rate(alternating, len(rows))
    human_verification_rate = _rate(human_verified, len(rows))
    duplicate_rate = 1.0 - len(set(fingerprints)) / len(fingerprints) if fingerprints else 0.0
    use_case_counts = dict(sorted(Counter(row.use_case for row in rows).items()))
    failures = []
    if len(rows) < limits.min_conversations:
        failures.append("conversation_count below threshold")
    if len(use_case_counts) < limits.min_use_cases:
        failures.append("use_case_count below threshold")
    if citation_valid_rate < limits.min_citation_valid_rate:
        failures.append("citation_valid_rate below threshold")
    if grounded_quote_rate < limits.min_grounded_quote_rate:
        failures.append("grounded_quote_rate below threshold")
    if role_alternation_rate < limits.min_role_alternation_rate:
        failures.append("role_alternation_rate below threshold")
    if human_verification_rate < limits.min_human_verification_rate:
        failures.append("human_verification_rate below threshold")
    if duplicate_rate > limits.max_duplicate_rate:
        failures.append("duplicate_rate above threshold")
    if pii_findings > limits.max_pii_findings:
        failures.append("pii_findings above threshold")
    return ConversationEvaluationReport(
        conversation_count=len(rows),
        turn_count=sum(len(row.turns) for row in rows),
        citation_count=citation_count,
        citation_valid_rate=citation_valid_rate,
        grounded_quote_rate=grounded_quote_rate,
        role_alternation_rate=role_alternation_rate,
        human_verification_rate=human_verification_rate,
        duplicate_rate=duplicate_rate,
        pii_findings=pii_findings,
        use_case_counts=use_case_counts,
        passed=bool(rows) and not failures,
        failures=failures or ([] if rows else ["dataset is empty"]),
    )


def write_conversations_jsonl(conversations: Iterable[SupportConversation], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for conversation in conversations:
            output.write(conversation.model_dump_json() + "\n")
            count += 1
    return count


def read_conversations_jsonl(path: Path) -> list[SupportConversation]:
    conversations = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                conversations.append(SupportConversation.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid conversation at {path}:{line_number}: {exc}") from exc
    return conversations


def conversations_to_safe_synth_frame(conversations: Iterable[SupportConversation]) -> pd.DataFrame:
    rows = []
    for conversation in conversations:
        citation = conversation.citations[0]
        rows.append(
            {
                "conversation_id": conversation.conversation_id,
                "use_case": conversation.use_case,
                "age_band": conversation.customer_profile.age_band,
                "state": conversation.customer_profile.state,
                "plan_type": conversation.customer_profile.plan_type,
                "channel": conversation.customer_profile.channel,
                "transcript": "\n".join(f"{turn.role}: {turn.content}" for turn in conversation.turns),
                "source_id": citation.source_id,
                "source_page": citation.page,
                "synthetic_only": True,
                "requires_human_verification": True,
            }
        )
    return pd.DataFrame(rows)


def _grounded_excerpt(text: str, max_characters: int = 600) -> str:
    excerpt = text.strip()[:max_characters].strip()
    if len(text.strip()) > max_characters and " " in excerpt:
        excerpt = excerpt.rsplit(" ", 1)[0]
    return excerpt


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
