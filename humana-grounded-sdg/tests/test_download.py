import hashlib
from pathlib import Path

import httpx
import pytest

from humana_sdg.download import DownloadError, download_source
from humana_sdg.manifest import load_manifest

MANIFEST = Path(__file__).parents[1] / "corpus" / "manifest.json"


def _source():
    return load_manifest(MANIFEST).sources[0]


def test_download_source_verifies_pdf_and_returns_audit_receipt(tmp_path: Path) -> None:
    body = b"%PDF-1.4\nsynthetic test document\n%%EOF"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "application/pdf"}, content=body)
    )

    with httpx.Client(transport=transport) as client:
        receipt = download_source(_source(), tmp_path, client=client)

    assert receipt.sha256 == hashlib.sha256(body).hexdigest()
    assert receipt.byte_count == len(body)
    assert receipt.path.read_bytes() == body
    assert receipt.source_id == _source().id


def test_download_source_rejects_non_pdf_without_leaving_file(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>no</html>")
    )

    with httpx.Client(transport=transport) as client:
        with pytest.raises(DownloadError, match="Content-Type"):
            download_source(_source(), tmp_path, client=client)

    assert list(tmp_path.iterdir()) == []
