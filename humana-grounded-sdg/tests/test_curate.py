from humana_sdg.curate import CuratorSettings, curate_chunks_python
from humana_sdg.models import GroundingChunk


def _chunk(chunk_id: str, text: str) -> GroundingChunk:
    return GroundingChunk(
        chunk_id=chunk_id,
        source_id="source",
        source_title="Title",
        source_url="https://www.cms.gov/reference.pdf",
        source_sha256="a" * 64,
        publisher="CMS",
        year=2026,
        category="claims_billing",
        page=1,
        text=text,
        citation_label="Title, p. 1",
    )


def test_python_postprocessing_filters_short_and_exact_duplicate_chunks() -> None:
    kept = "Providers must verify eligibility before delivering non-emergency covered services to a member."
    chunks = [
        _chunk("one", kept),
        _chunk("two", "  " + kept.upper() + "  "),
        _chunk("three", "too short"),
    ]

    curated = curate_chunks_python(chunks, CuratorSettings(min_words=8, max_words=30))

    assert [chunk.chunk_id for chunk in curated] == ["one"]


def test_python_postprocessing_is_deterministic() -> None:
    chunks = [
        _chunk("b", "This source provides enough words for a deterministic curation record."),
        _chunk("a", "Another source provides enough words for a second deterministic record."),
    ]

    first = curate_chunks_python(chunks, CuratorSettings(min_words=8, max_words=30))
    second = curate_chunks_python(list(reversed(chunks)), CuratorSettings(min_words=8, max_words=30))

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
