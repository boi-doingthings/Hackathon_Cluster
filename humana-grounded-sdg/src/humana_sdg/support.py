from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from humana_sdg.evaluate import _count_pii
from humana_sdg.models import Citation, GroundingChunk

UseCase = Literal[
    "appeals-grievances",
    "care-management",
    "claims-billing",
    "compliance-reporting",
    "dental-benefits",
    "eligibility-enrollment",
    "medicaid-member-support",
    "pharmacy-benefits",
    "plan-benefits",
    "prior-authorization",
    "privacy-preferences",
    "provider-directory",
]
Channel = Literal["chat", "phone-transcript", "secure-message"]
CustomerPersona = Literal["member", "caregiver", "provider-office", "prospective-member"]
CustomerSentiment = Literal["neutral", "confused", "concerned", "frustrated"]
Outcome = Literal["information-provided", "human-escalation", "current-document-referral"]

SUPPORT_DISCLAIMER = (
    "Fully synthetic customer-support conversation grounded only in the cited public document; "
    "not a coverage determination, medical advice, claim decision, or eligibility decision."
)

_USE_CASES_BY_CATEGORY: dict[str, tuple[UseCase, ...]] = {
    "appeals_grievances": ("appeals-grievances",),
    "claims_billing": ("claims-billing",),
    "compliance": ("compliance-reporting",),
    "corporate_impact": ("care-management",),
    "coverage_policy": ("plan-benefits", "care-management"),
    "dental_provider_operations": ("dental-benefits", "provider-directory"),
    "evidence_of_coverage": ("plan-benefits", "eligibility-enrollment"),
    "medicaid_provider_manual": ("medicaid-member-support", "care-management", "eligibility-enrollment"),
    "medicare_reference": ("plan-benefits", "eligibility-enrollment"),
    "pharmacy": ("pharmacy-benefits",),
    "prior_authorization": ("prior-authorization",),
    "privacy": ("privacy-preferences",),
    "provider_operations": ("provider-directory", "claims-billing", "prior-authorization"),
}

_CUSTOMER_GOALS: dict[UseCase, tuple[str, ...]] = {
    "appeals-grievances": (
        "I need to understand the documented appeal or grievance process.",
        "I want to know where the public guide describes next steps after a disputed decision.",
    ),
    "care-management": (
        "I am looking for the documented care-support resources described by the plan.",
        "I need help understanding how to verify available care-management support.",
    ),
    "claims-billing": (
        "I need help understanding a billing or claim-status process without sharing real claim data.",
        "I want to know which documented billing guidance applies before contacting support.",
    ),
    "compliance-reporting": (
        "I want to understand how the public policy says a concern should be reported.",
        "I need the documented compliance route for a synthetic support scenario.",
    ),
    "dental-benefits": (
        "I need to understand documented dental-office or dental-benefit guidance.",
        "I want to verify the public dental guidance before scheduling a service.",
    ),
    "eligibility-enrollment": (
        "I need general enrollment or eligibility guidance and know this chat cannot decide eligibility.",
        "I want to find the current source used to verify enrollment-related requirements.",
    ),
    "medicaid-member-support": (
        "I need the public Medicaid-plan guidance for a member-support question.",
        "I want to verify which state-plan document should guide my next support step.",
    ),
    "pharmacy-benefits": (
        "I need help understanding documented pharmacy or step-therapy guidance.",
        "I want to know where the current drug-related requirement can be verified.",
    ),
    "plan-benefits": (
        "I need a plain-language explanation of the cited plan-benefit guidance.",
        "I want to verify a benefit rule without treating this transcript as a coverage decision.",
    ),
    "prior-authorization": (
        "I need to understand the documented prior-authorization process for a synthetic case.",
        "I want to verify whether the public source lists an authorization-related requirement.",
    ),
    "privacy-preferences": (
        "I want to understand the public privacy policy without sharing personal information.",
        "I need the documented route for a privacy preference or concern.",
    ),
    "provider-directory": (
        "I need guidance on verifying a provider through current official resources.",
        "I want to understand the documented provider-support process.",
    ),
}

_PERSONAS: tuple[CustomerPersona, ...] = ("member", "caregiver", "provider-office", "prospective-member")
_CHANNELS: tuple[Channel, ...] = ("chat", "phone-transcript", "secure-message")
_SENTIMENTS: tuple[CustomerSentiment, ...] = ("neutral", "confused", "concerned", "frustrated")
_OUTCOMES: tuple[Outcome, ...] = (
    "information-provided",
    "human-escalation",
    "current-document-referral",
)
_UNSAFE_AGENT_PATTERNS = (
    re.compile(r"\bguarantee(?:d|s)?\s+(?:approval|coverage|payment|eligibility)\b", re.IGNORECASE),
    re.compile(r"\byou are\s+(?:covered|eligible|approved)\b", re.IGNORECASE),
)


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["customer", "agent"]
    content: str = Field(min_length=1)


class SupportConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(pattern=r"^SYN-CONV-[0-9A-F]{16}$")
    use_case: UseCase
    channel: Channel
    customer_persona: CustomerPersona
    customer_sentiment: CustomerSentiment
    outcome: Outcome
    turns: list[ConversationTurn] = Field(min_length=6, max_length=10)
    citations: list[Citation] = Field(min_length=1)
    is_synthetic: Literal[True] = True
    requires_human_verification: Literal[True] = True
    disclaimer: str = SUPPORT_DISCLAIMER


class SupportEvaluationThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_citation_valid_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_grounded_quote_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_turn_structure_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_safety_compliance_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    max_duplicate_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    max_pii_findings: int = Field(default=0, ge=0)
    min_use_cases: int = Field(default=10, ge=1)


class SupportEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_count: int
    turn_count: int
    average_turn_count: float
    citation_count: int
    citation_valid_rate: float
    grounded_quote_rate: float
    turn_structure_rate: float
    safety_compliance_rate: float
    safety_violation_count: int
    duplicate_rate: float
    pii_findings: int
    use_case_count: int
    use_case_counts: dict[str, int]
    channel_counts: dict[str, int]
    persona_counts: dict[str, int]
    outcome_counts: dict[str, int]
    passed: bool
    failures: list[str]


def generate_support_conversations(
    chunks: Iterable[GroundingChunk],
    *,
    count: int = 1200,
    seed: int = 20260722,
) -> list[SupportConversation]:
    if count < 1:
        raise ValueError("count must be at least 1")
    chunk_list = sorted(chunks, key=lambda item: item.chunk_id)
    if not chunk_list:
        raise ValueError("At least one grounding chunk is required")

    buckets: dict[UseCase, list[GroundingChunk]] = defaultdict(list)
    for chunk in chunk_list:
        mapped = _USE_CASES_BY_CATEGORY.get(chunk.category, ("plan-benefits",))
        for use_case in mapped:
            buckets[use_case].append(chunk)

    rng = random.Random(seed)  # noqa: S311 - deterministic synthetic-data sampling
    for bucket in buckets.values():
        rng.shuffle(bucket)
    use_cases = sorted(buckets)
    conversations = []
    for index in range(count):
        use_case = use_cases[index % len(use_cases)]
        cycle = index // len(use_cases)
        bucket = buckets[use_case]
        chunk = bucket[cycle % len(bucket)]
        variant = cycle // len(bucket)
        conversations.append(_build_conversation(chunk, use_case, index, variant, seed))
    return conversations


def evaluate_support_conversations(
    conversations: Iterable[SupportConversation],
    chunks: Iterable[GroundingChunk],
    thresholds: SupportEvaluationThresholds | None = None,
) -> SupportEvaluationReport:
    limits = thresholds or SupportEvaluationThresholds()
    items = list(conversations)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    citation_count = valid_citations = grounded_quotes = 0
    valid_structure = safe_conversations = pii_findings = turn_count = 0
    fingerprints = []

    for conversation in items:
        turn_count += len(conversation.turns)
        valid_structure += int(_has_valid_turn_structure(conversation))
        agent_text = "\n".join(turn.content for turn in conversation.turns if turn.role == "agent")
        all_text = "\n".join(turn.content for turn in conversation.turns)
        pii_findings += _count_pii(all_text)
        safe_conversations += int(_is_safe_conversation(conversation, agent_text))
        for citation in conversation.citations:
            citation_count += 1
            chunk = chunk_by_id.get(citation.chunk_id)
            metadata_matches = chunk is not None and (
                chunk.source_id,
                chunk.page,
                chunk.source_url,
            ) == (
                citation.source_id,
                citation.page,
                citation.source_url,
            )
            if metadata_matches:
                valid_citations += 1
                quote_grounded = _normalize(citation.quote) in _normalize(chunk.text)
                grounded_quotes += int(quote_grounded and citation.quote in agent_text)
        fingerprints.append(hashlib.sha256(_normalize(all_text).encode()).hexdigest())

    count = len(items)
    citation_valid_rate = _rate(valid_citations, citation_count)
    grounded_quote_rate = _rate(grounded_quotes, citation_count)
    turn_structure_rate = _rate(valid_structure, count)
    safety_compliance_rate = _rate(safe_conversations, count)
    duplicate_rate = 1.0 - len(set(fingerprints)) / len(fingerprints) if fingerprints else 0.0
    use_case_counts = Counter(item.use_case for item in items)
    failures = []
    if citation_valid_rate < limits.min_citation_valid_rate:
        failures.append("citation_valid_rate below threshold")
    if grounded_quote_rate < limits.min_grounded_quote_rate:
        failures.append("grounded_quote_rate below threshold")
    if turn_structure_rate < limits.min_turn_structure_rate:
        failures.append("turn_structure_rate below threshold")
    if safety_compliance_rate < limits.min_safety_compliance_rate:
        failures.append("safety_compliance_rate below threshold")
    if duplicate_rate > limits.max_duplicate_rate:
        failures.append("duplicate_rate above threshold")
    if pii_findings > limits.max_pii_findings:
        failures.append("pii_findings above threshold")
    if len(use_case_counts) < limits.min_use_cases:
        failures.append("use_case_count below threshold")
    if not items:
        failures.append("dataset is empty")

    return SupportEvaluationReport(
        conversation_count=count,
        turn_count=turn_count,
        average_turn_count=_rate(turn_count, count),
        citation_count=citation_count,
        citation_valid_rate=citation_valid_rate,
        grounded_quote_rate=grounded_quote_rate,
        turn_structure_rate=turn_structure_rate,
        safety_compliance_rate=safety_compliance_rate,
        safety_violation_count=count - safe_conversations,
        duplicate_rate=duplicate_rate,
        pii_findings=pii_findings,
        use_case_count=len(use_case_counts),
        use_case_counts=dict(use_case_counts),
        channel_counts=dict(Counter(item.channel for item in items)),
        persona_counts=dict(Counter(item.customer_persona for item in items)),
        outcome_counts=dict(Counter(item.outcome for item in items)),
        passed=not failures,
        failures=failures,
    )


def write_support_datasets(
    conversations: Iterable[SupportConversation], output_directory: Path
) -> dict[str, Path]:
    items = list(conversations)
    output_directory.mkdir(parents=True, exist_ok=True)
    conversations_path = output_directory / "support_conversations.jsonl"
    sft_path = output_directory / "support_sft_openai.jsonl"
    with conversations_path.open("w", encoding="utf-8", newline="\n") as output:
        for conversation in items:
            output.write(conversation.model_dump_json() + "\n")
    with sft_path.open("w", encoding="utf-8", newline="\n") as output:
        for conversation in items:
            payload = {
                "messages": [
                    {
                        "role": "user" if turn.role == "customer" else "assistant",
                        "content": turn.content,
                    }
                    for turn in conversation.turns
                ],
                "metadata": {
                    "conversation_id": conversation.conversation_id,
                    "use_case": conversation.use_case,
                    "channel": conversation.channel,
                    "customer_persona": conversation.customer_persona,
                    "outcome": conversation.outcome,
                    "citations": [citation.model_dump(mode="json") for citation in conversation.citations],
                    "is_synthetic": True,
                    "requires_human_verification": True,
                    "disclaimer": conversation.disclaimer,
                },
            }
            output.write(json.dumps(payload, separators=(",", ":")) + "\n")
    return {"conversations": conversations_path, "openai_sft": sft_path}


def read_support_conversations_jsonl(path: Path) -> list[SupportConversation]:
    items = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                items.append(SupportConversation.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid support conversation at {path}:{line_number}: {exc}") from exc
    return items


def _build_conversation(
    chunk: GroundingChunk,
    use_case: UseCase,
    index: int,
    variant: int,
    seed: int,
) -> SupportConversation:
    identity = f"{seed}|{index}|{variant}|{use_case}|{chunk.chunk_id}".encode()
    digest = hashlib.sha256(identity).hexdigest().upper()
    conversation_id = f"SYN-CONV-{digest[:16]}"
    case_reference = f"SYN-CASE-{digest[16:28]}"
    persona = _PERSONAS[index % len(_PERSONAS)]
    channel = _CHANNELS[index % len(_CHANNELS)]
    sentiment = _SENTIMENTS[(index + variant) % len(_SENTIMENTS)]
    outcome = _OUTCOMES[(index + variant) % len(_OUTCOMES)]
    goals = _CUSTOMER_GOALS[use_case]
    goal = goals[(index + variant) % len(goals)]
    quote = _grounded_excerpt(chunk.text)
    turns = [
        ConversationTurn(
            role="customer",
            content=(
                f"I am contacting support as a {persona} with synthetic reference "
                f"{case_reference}. {goal}"
            ),
        ),
        ConversationTurn(
            role="agent",
            content=(
                "I can explain the cited public guidance and safe next steps, but I cannot make a "
                "coverage, eligibility, claim, authorization, payment, or medical determination."
            ),
        ),
        ConversationTurn(
            role="customer",
            content="What does the official source say, and where can I verify it?",
        ),
        ConversationTurn(
            role="agent",
            content=f"{chunk.source_title}, page {chunk.page}, states: {quote}",
        ),
        ConversationTurn(
            role="customer",
            content="Does that statement guarantee the outcome for this synthetic case?",
        ),
        ConversationTurn(
            role="agent",
            content=(
                "No. Public guidance can change and plan-specific facts matter. Please do not provide PHI, "
                "member identifiers, or medical records in this training transcript."
            ),
        ),
        ConversationTurn(
            role="customer",
            content="What is the safest next step if I still need help?",
        ),
        ConversationTurn(
            role="agent",
            content=_closing_message(outcome),
        ),
    ]
    return SupportConversation(
        conversation_id=conversation_id,
        use_case=use_case,
        channel=channel,
        customer_persona=persona,
        customer_sentiment=sentiment,
        outcome=outcome,
        turns=turns,
        citations=[Citation(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            source_url=chunk.source_url,
            page=chunk.page,
            quote=quote,
        )],
        is_synthetic=True,
        requires_human_verification=True,
        disclaimer=SUPPORT_DISCLAIMER,
    )


def _closing_message(outcome: Outcome) -> str:
    if outcome == "human-escalation":
        action = "Contact an authorized human representative through the official support channel."
    elif outcome == "current-document-referral":
        action = "Use the current plan document and official Humana or CMS lookup resource."
    else:
        action = (
            "Keep the cited page for reference and use the official support channel "
            "for case-specific help."
        )
    return (
        "This is not a coverage determination. Verify the current plan document or official source. "
        f"{action} A human representative must confirm any case-specific outcome."
    )


def _grounded_excerpt(text: str, max_characters: int = 600) -> str:
    excerpt = text.strip()[:max_characters].strip()
    if len(text.strip()) > max_characters and " " in excerpt:
        excerpt = excerpt.rsplit(" ", 1)[0]
    return excerpt


def _has_valid_turn_structure(conversation: SupportConversation) -> bool:
    return (
        6 <= len(conversation.turns) <= 10
        and conversation.turns[0].role == "customer"
        and conversation.turns[-1].role == "agent"
        and all(
            left.role != right.role
            for left, right in zip(conversation.turns, conversation.turns[1:], strict=False)
        )
    )


def _is_safe_conversation(conversation: SupportConversation, agent_text: str) -> bool:
    final_agent = conversation.turns[-1].content.casefold()
    required = (
        conversation.is_synthetic,
        conversation.requires_human_verification,
        "not a coverage determination" in conversation.disclaimer.casefold(),
        "not a coverage determination" in final_agent,
        "verify" in final_agent,
        "human representative" in final_agent,
    )
    # Official excerpts can contain contextual language such as "you are covered" or
    # explicit warnings that an ID card does not guarantee coverage. They are exact,
    # validated citations and must not be mistaken for an uncited agent determination.
    uncited_agent_text = agent_text
    for citation in conversation.citations:
        uncited_agent_text = uncited_agent_text.replace(citation.quote, "")
    return all(required) and not any(
        pattern.search(uncited_agent_text) for pattern in _UNSAFE_AGENT_PATTERNS
    )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
