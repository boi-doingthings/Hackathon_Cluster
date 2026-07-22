from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from humana_sdg.models import Citation, GroundingChunk, SyntheticRecord

DISCLAIMER = (
    "Synthetic training example grounded only in the cited public document; not a coverage determination, "
    "medical advice, or a substitute for the current plan document."
)
QUESTION_TEMPLATES = (
    "According to {title} page {page}, what does the cited passage say about {topic}?",
    "Using only {title} page {page}, summarize the relevant guidance about {topic}.",
    "What grounded operational guidance can be taken from {title} page {page} about {topic}?",
)
TASK_BY_CATEGORY = {
    "prior_authorization": "prior_authorization",
    "pharmacy": "prior_authorization",
    "claims_billing": "claims_billing",
    "provider_operations": "claims_billing",
    "dental_provider_operations": "claims_billing",
    "appeals_grievances": "appeals_grievances",
    "evidence_of_coverage": "coverage_explanation",
    "coverage_policy": "coverage_explanation",
}


def generate_deterministic_records(
    chunks: Iterable[GroundingChunk],
    *,
    records_per_chunk: int = 1,
) -> list[SyntheticRecord]:
    if not 1 <= records_per_chunk <= len(QUESTION_TEMPLATES):
        raise ValueError(f"records_per_chunk must be between 1 and {len(QUESTION_TEMPLATES)}")

    records: list[SyntheticRecord] = []
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        quote = _grounded_excerpt(chunk.text)
        topic = chunk.category.replace("_", " ")
        for variant in range(records_per_chunk):
            question = QUESTION_TEMPLATES[variant].format(
                title=chunk.source_title,
                page=chunk.page,
                topic=topic,
            )
            answer = f"The cited passage states: {quote}"
            record_id = _record_id(chunk.chunk_id, variant, question, answer)
            records.append(
                SyntheticRecord(
                    record_id=record_id,
                    task_type=TASK_BY_CATEGORY.get(chunk.category, "grounded_qa"),
                    question=question,
                    answer=answer,
                    citations=[
                        Citation(
                            chunk_id=chunk.chunk_id,
                            source_id=chunk.source_id,
                            source_url=chunk.source_url,
                            page=chunk.page,
                            quote=quote,
                        )
                    ],
                    is_synthetic=True,
                    disclaimer=DISCLAIMER,
                )
            )
    return records


def write_records_jsonl(records: Iterable[SyntheticRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(record.model_dump_json() + "\n")
            count += 1
    return count


def _grounded_excerpt(text: str, max_characters: int = 700) -> str:
    excerpt = text.strip()[:max_characters].strip()
    if len(text.strip()) > max_characters and " " in excerpt:
        excerpt = excerpt.rsplit(" ", 1)[0]
    return excerpt


def _record_id(chunk_id: str, variant: int, question: str, answer: str) -> str:
    payload = f"{chunk_id}\n{variant}\n{question}\n{answer}".encode()
    return hashlib.sha256(payload).hexdigest()
