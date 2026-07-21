from pathlib import Path

import pytest

from humana_sdg.manifest import ManifestError, load_manifest, validate_source_url

MANIFEST = Path(__file__).parents[1] / "corpus" / "manifest.json"


def test_manifest_is_large_unique_and_pdf_only() -> None:
    manifest = load_manifest(MANIFEST)

    source_ids = [source.id for source in manifest.sources]
    assert len(source_ids) == 22
    assert len(source_ids) == len(set(source_ids))
    assert all(source.verified_content_type == "application/pdf" for source in manifest.sources)


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/member.pdf",
        "http://assets.humana.com/is/content/humana/filepdf",
        "https://assets.humana.com.evil.example/file.pdf",
    ],
)
def test_source_url_rejects_untrusted_or_insecure_hosts(url: str) -> None:
    with pytest.raises(ManifestError):
        validate_source_url(url)


def test_source_url_accepts_official_publishers() -> None:
    validate_source_url("https://assets.humana.com/is/content/humana/filepdf")
    validate_source_url("https://www.cms.gov/files/document/reference.pdf")
    validate_source_url("https://www.medicare.gov/publications/reference.pdf")
