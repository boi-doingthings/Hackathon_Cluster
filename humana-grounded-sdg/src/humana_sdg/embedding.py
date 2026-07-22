from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from humana_sdg.models import GroundingChunk


class EmbeddingTriplet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    pos_doc: str
    neg_doc: list[str] = Field(min_length=1)


def build_embedding_triplets(
    chunks: Iterable[GroundingChunk],
    *,
    seed: int = 42,
) -> list[EmbeddingTriplet]:
    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    if len(ordered) < 2:
        raise ValueError("At least two chunks are required to form embedding triplets")

    rng = random.Random(seed)
    triplets: list[EmbeddingTriplet] = []
    for chunk in ordered:
        candidates = [candidate for candidate in ordered if candidate.source_id != chunk.source_id]
        if not candidates:
            candidates = [candidate for candidate in ordered if candidate.chunk_id != chunk.chunk_id]
        negative = rng.choice(candidates)
        triplets.append(
            EmbeddingTriplet(
                query=(
                    f"What guidance does {chunk.source_title} page {chunk.page} provide about "
                    f"{chunk.category.replace('_', ' ')}?"
                ),
                pos_doc=chunk.text,
                neg_doc=[negative.text],
            )
        )
    return triplets


def split_and_write_triplets(
    triplets: Iterable[EmbeddingTriplet],
    output_directory: Path,
    *,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Path, Path]:
    items = list(triplets)
    if len(items) < 2:
        raise ValueError("At least two triplets are required for train/validation output")
    if not 0.01 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0.01, 0.5)")

    random.Random(seed).shuffle(items)
    validation_size = max(1, round(len(items) * validation_fraction))
    validation = items[:validation_size]
    training = items[validation_size:]
    output_directory.mkdir(parents=True, exist_ok=True)
    training_path = output_directory / "training.jsonl"
    validation_path = output_directory / "validation.jsonl"
    _write_jsonl(training, training_path)
    _write_jsonl(validation, validation_path)
    return training_path, validation_path


def _write_jsonl(items: Iterable[EmbeddingTriplet], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for item in items:
            output.write(item.model_dump_json() + "\n")
