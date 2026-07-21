from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from humana_sdg.models import GroundingChunk, SyntheticRecord

TOKEN = re.compile(r"[a-z0-9]+")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@([\w.-]+\.[A-Za-z]{2,})\b")
PUBLIC_CONTACT_DOMAINS = {
    "aetna.com",
    "cms.gov",
    "cms.hhs.gov",
    "hhs.gov",
    "humana.com",
    "okhca.org",
}
PII_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b[A-Z0-9]{4}-[A-Z0-9]{3}-[A-Z0-9]{4}\b", re.IGNORECASE),
    re.compile(r"\b(?:date of birth|dob)\s*[:=-]\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
)


class EvaluationThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_citation_valid_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_grounded_quote_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_answer_support_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    max_duplicate_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    max_pii_findings: int = Field(default=0, ge=0)


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int
    citation_count: int
    citation_valid_rate: float
    grounded_quote_rate: float
    answer_support_rate: float
    duplicate_rate: float
    pii_findings: int
    task_type_counts: dict[str, int]
    passed: bool
    failures: list[str]


def evaluate_records(
    records: Iterable[SyntheticRecord],
    chunks: Iterable[GroundingChunk],
    thresholds: EvaluationThresholds,
) -> EvaluationReport:
    record_list = list(records)
    chunk_list = list(chunks)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunk_list}

    citation_count = 0
    valid_citations = 0
    grounded_quotes = 0
    supported_answers = 0
    pii_findings = 0
    fingerprints: list[str] = []

    for record in record_list:
        record_has_support = False
        for citation in record.citations:
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
                if _normalize(citation.quote) in _normalize(chunk.text):
                    grounded_quotes += 1
            if _answer_supported_by_quote(record.answer, citation.quote):
                record_has_support = True
        supported_answers += int(record_has_support)
        pii_findings += _count_pii(f"{record.question}\n{record.answer}")
        fingerprints.append(_fingerprint(record.question, record.answer))

    citation_valid_rate = _rate(valid_citations, citation_count)
    grounded_quote_rate = _rate(grounded_quotes, citation_count)
    answer_support_rate = _rate(supported_answers, len(record_list))
    duplicate_rate = _duplicate_rate(fingerprints)
    failures = _threshold_failures(
        citation_valid_rate,
        grounded_quote_rate,
        answer_support_rate,
        duplicate_rate,
        pii_findings,
        thresholds,
    )
    return EvaluationReport(
        record_count=len(record_list),
        citation_count=citation_count,
        citation_valid_rate=citation_valid_rate,
        grounded_quote_rate=grounded_quote_rate,
        answer_support_rate=answer_support_rate,
        duplicate_rate=duplicate_rate,
        pii_findings=pii_findings,
        task_type_counts=dict(Counter(record.task_type for record in record_list)),
        passed=bool(record_list) and not failures,
        failures=failures or ([] if record_list else ["dataset is empty"]),
    )


def _answer_supported_by_quote(answer: str, quote: str) -> bool:
    normalized_answer = _normalize(answer)
    normalized_quote = _normalize(quote)
    if normalized_quote and normalized_quote in normalized_answer:
        return True
    answer_tokens = set(TOKEN.findall(normalized_answer))
    quote_tokens = set(TOKEN.findall(normalized_quote))
    return bool(quote_tokens) and len(answer_tokens & quote_tokens) / len(quote_tokens) >= 0.8


def _is_public_contact_domain(domain: str) -> bool:
    normalized = domain.casefold()
    return normalized in PUBLIC_CONTACT_DOMAINS or normalized.endswith(".gov")


def _count_pii(text: str) -> int:
    sensitive_emails = sum(
        1 for match in EMAIL_PATTERN.finditer(text) if not _is_public_contact_domain(match.group(1))
    )
    return sensitive_emails + sum(len(pattern.findall(text)) for pattern in PII_PATTERNS)


def _fingerprint(question: str, answer: str) -> str:
    return hashlib.sha256(f"{_normalize(question)}\n{_normalize(answer)}".encode()).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _duplicate_rate(fingerprints: list[str]) -> float:
    return 1.0 - len(set(fingerprints)) / len(fingerprints) if fingerprints else 0.0


def _threshold_failures(
    citation_valid_rate: float,
    grounded_quote_rate: float,
    answer_support_rate: float,
    duplicate_rate: float,
    pii_findings: int,
    thresholds: EvaluationThresholds,
) -> list[str]:
    failures = []
    if citation_valid_rate < thresholds.min_citation_valid_rate:
        failures.append("citation_valid_rate below threshold")
    if grounded_quote_rate < thresholds.min_grounded_quote_rate:
        failures.append("grounded_quote_rate below threshold")
    if answer_support_rate < thresholds.min_answer_support_rate:
        failures.append("answer_support_rate below threshold")
    if duplicate_rate > thresholds.max_duplicate_rate:
        failures.append("duplicate_rate above threshold")
    if pii_findings > thresholds.max_pii_findings:
        failures.append("pii_findings above threshold")
    return failures
