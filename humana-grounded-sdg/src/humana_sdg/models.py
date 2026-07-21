from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GroundingChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    source_id: str
    source_title: str
    source_url: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publisher: str
    year: int
    category: str
    page: int = Field(ge=1)
    text: str
    citation_label: str


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    source_id: str
    source_url: str
    page: int = Field(ge=1)
    quote: str


class SyntheticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    task_type: Literal[
        "grounded_qa",
        "coverage_explanation",
        "prior_authorization",
        "claims_billing",
        "appeals_grievances",
        "tool_calling",
    ]
    question: str
    answer: str
    citations: list[Citation] = Field(min_length=1)
    is_synthetic: Literal[True] = True
    disclaimer: str
