from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import fitz

from humana_sdg.manifest import GroundingSource
from humana_sdg.models import GroundingChunk

WHITESPACE = re.compile(r"\s+")


def normalize_extracted_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", normalized)
    return WHITESPACE.sub(" ", normalized).strip()


def extract_pdf(
    source: GroundingSource,
    pdf_path: Path,
    *,
    pdf_sha256: str,
    max_words: int = 220,
    overlap_words: int = 30,
) -> list[GroundingChunk]:
    if max_words < 1 or not 0 <= overlap_words < max_words:
        raise ValueError("Require max_words >= 1 and 0 <= overlap_words < max_words")

    chunks: list[GroundingChunk] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_text = normalize_extracted_text(page.get_text("text"))
            if not page_text:
                continue
            for chunk_index, text in enumerate(_word_windows(page_text, max_words, overlap_words)):
                chunk_id = _chunk_id(source.id, pdf_sha256, page_index + 1, chunk_index, text)
                chunks.append(
                    GroundingChunk(
                        chunk_id=chunk_id,
                        source_id=source.id,
                        source_title=source.title,
                        source_url=source.url,
                        source_sha256=pdf_sha256,
                        publisher=source.publisher,
                        year=source.year,
                        category=source.category,
                        page=page_index + 1,
                        text=text,
                        citation_label=f"{source.title}, p. {page_index + 1}",
                    )
                )
    return chunks


def _word_windows(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    step = max_words - overlap_words
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), step)]


def _chunk_id(source_id: str, pdf_sha256: str, page: int, chunk_index: int, text: str) -> str:
    payload = f"{source_id}\n{pdf_sha256}\n{page}\n{chunk_index}\n{text}".encode()
    return hashlib.sha256(payload).hexdigest()
