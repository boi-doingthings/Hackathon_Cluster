from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from humana_sdg.models import GroundingChunk


class CuratorUnavailable(RuntimeError):
    """Raised when the NVIDIA NeMo Curator runtime is not installed."""


class CuratorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_words: int = Field(default=40, ge=1)
    max_words: int = Field(default=350, ge=1)


class CuratorRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_path: Path
    output_directory: Path
    output_files: list[Path]
    nemo_curator_version: str


def curate_chunks_python(
    chunks: Iterable[GroundingChunk],
    settings: CuratorSettings,
) -> list[GroundingChunk]:
    if settings.min_words > settings.max_words:
        raise ValueError("min_words cannot exceed max_words")

    unique: dict[str, GroundingChunk] = {}
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        word_count = len(chunk.text.split())
        if not settings.min_words <= word_count <= settings.max_words:
            continue
        unique.setdefault(_canonical_text(chunk.text), chunk)
    return sorted(unique.values(), key=lambda item: item.chunk_id)


def write_chunks_jsonl(chunks: Iterable[GroundingChunk], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(chunk.model_dump_json() + "\n")
            count += 1
    return count


def read_chunks_jsonl(paths: Path | Iterable[Path]) -> list[GroundingChunk]:
    input_paths = [paths] if isinstance(paths, Path) else sorted(paths)
    fields = set(GroundingChunk.model_fields)
    chunks: list[GroundingChunk] = []
    for path in input_paths:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    chunks.append(GroundingChunk.model_validate({key: payload[key] for key in fields}))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise ValueError(f"Invalid chunk at {path}:{line_number}: {exc}") from exc
    return chunks


def run_nemo_curator(
    input_jsonl: Path,
    output_directory: Path,
    settings: CuratorSettings,
) -> CuratorRun:
    """Run NeMo Curator 1.2 Pipeline before downstream data generation."""
    try:
        import nemo_curator
        from nemo_curator.pipeline import Pipeline
        from nemo_curator.stages.text.filters import ScoreFilter
        from nemo_curator.stages.text.filters.heuristic import WordCountFilter
        from nemo_curator.stages.text.io.reader import JsonlReader
        from nemo_curator.stages.text.io.writer import JsonlWriter
        from nemo_curator.stages.text.modules import AddId
    except ImportError as exc:
        raise CuratorUnavailable("Install the Curator extra first: uv sync --extra curator") from exc

    if settings.min_words > settings.max_words:
        raise ValueError("min_words cannot exceed max_words")

    output_directory.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(
        name="humana-grounding-curation",
        description="Filter page-grounded Humana/CMS chunks before any synthetic generation.",
    )
    pipeline.add_stage(JsonlReader(file_paths=str(input_jsonl), files_per_partition=1))
    pipeline.add_stage(AddId(id_field="curator_id", id_prefix="humana_grounding"))
    pipeline.add_stage(
        ScoreFilter(
            filter_obj=WordCountFilter(
                min_words=settings.min_words,
                max_words=settings.max_words,
                lang="en",
            ),
            text_field="text",
            score_field="word_count",
        )
    )
    pipeline.add_stage(JsonlWriter(path=str(output_directory), mode="overwrite"))
    pipeline.run()

    output_files = sorted(output_directory.glob("*.jsonl"))
    if not output_files:
        raise RuntimeError("NeMo Curator completed without producing JSONL output")
    version = getattr(nemo_curator, "__version__", "unknown")
    return CuratorRun(
        input_path=input_jsonl,
        output_directory=output_directory,
        output_files=output_files,
        nemo_curator_version=version,
    )


def _canonical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()
