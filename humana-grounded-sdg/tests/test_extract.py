from pathlib import Path

import fitz

from humana_sdg.extract import extract_pdf
from humana_sdg.manifest import load_manifest

MANIFEST = Path(__file__).parents[1] / "corpus" / "manifest.json"


def test_extract_pdf_preserves_page_level_citations(tmp_path: Path) -> None:
    pdf_path = tmp_path / "grounding.pdf"
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "Prior authorization may be required before selected services are provided.")
    second = document.new_page()
    second.insert_text((72, 72), "An appeal asks the plan to review an adverse organization determination.")
    document.save(pdf_path)
    document.close()

    source = load_manifest(MANIFEST).sources[0]
    chunks = extract_pdf(source, pdf_path, pdf_sha256="a" * 64, max_words=100, overlap_words=10)

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert all(chunk.source_id == source.id for chunk in chunks)
    assert all(chunk.source_url == source.url for chunk in chunks)
    assert chunks[0].citation_label.endswith("p. 1")
    assert "Prior authorization" in chunks[0].text
    assert len({chunk.chunk_id for chunk in chunks}) == 2
