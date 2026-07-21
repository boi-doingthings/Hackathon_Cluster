from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

from humana_sdg.manifest import GroundingSource, validate_source_url

MAX_PDF_BYTES = 100 * 1024 * 1024


class DownloadError(RuntimeError):
    """Raised when a source cannot be safely downloaded as a PDF."""


class DownloadReceipt(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_id: str
    url: str
    path: Path
    sha256: str
    byte_count: int
    content_type: str
    retrieved_at: str


def download_source(
    source: GroundingSource,
    destination: Path,
    *,
    client: httpx.Client | None = None,
) -> DownloadReceipt:
    validate_source_url(source.url)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{source.id}.pdf"
    temporary = destination / f".{source.id}.pdf.part"

    owns_client = client is None
    http_client = client or httpx.Client(follow_redirects=True, timeout=90.0)
    try:
        return _download_with_client(source, target, temporary, http_client)
    finally:
        temporary.unlink(missing_ok=True)
        if owns_client:
            http_client.close()


def _download_with_client(
    source: GroundingSource,
    target: Path,
    temporary: Path,
    client: httpx.Client,
) -> DownloadReceipt:
    digest = hashlib.sha256()
    byte_count = 0

    try:
        with client.stream("GET", source.url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/pdf":
                raise DownloadError(f"Unexpected Content-Type for {source.id}: {content_type or 'missing'}")

            with temporary.open("wb") as output:
                for chunk in response.iter_bytes():
                    byte_count += len(chunk)
                    if byte_count > MAX_PDF_BYTES:
                        raise DownloadError(f"PDF exceeds {MAX_PDF_BYTES} bytes: {source.id}")
                    digest.update(chunk)
                    output.write(chunk)
    except (httpx.HTTPError, OSError) as exc:
        raise DownloadError(f"Failed to download {source.id}: {exc}") from exc

    if temporary.read_bytes()[:5] != b"%PDF-":
        raise DownloadError(f"Downloaded payload is not a PDF: {source.id}")

    temporary.replace(target)
    return DownloadReceipt(
        source_id=source.id,
        url=source.url,
        path=target,
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        content_type="application/pdf",
        retrieved_at=datetime.now(UTC).isoformat(),
    )
