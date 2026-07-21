from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

TRUSTED_SOURCE_HOSTS = frozenset(
    {
        "assets.humana.com",
        "humana.gcs-web.com",
        "cms.gov",
        "www.cms.gov",
        "medicare.gov",
        "www.medicare.gov",
    }
)


class ManifestError(ValueError):
    """Raised when a grounding manifest is unsafe or malformed."""


class GroundingSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    publisher: str
    year: int = Field(ge=2000, le=2100)
    category: str
    url: str
    why_useful: str
    rights_profile: str
    verified_at: str
    verified_content_type: str


class GroundingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: str
    description: str
    rights_profiles: dict[str, str]
    sources: list[GroundingSource]


def validate_source_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ManifestError(f"Grounding URL must use HTTPS: {url}")
    if parsed.hostname not in TRUSTED_SOURCE_HOSTS:
        raise ManifestError(f"Grounding URL host is not allow-listed: {parsed.hostname}")


def load_manifest(path: Path) -> GroundingManifest:
    try:
        manifest = GroundingManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot load grounding manifest {path}: {exc}") from exc

    source_ids = [source.id for source in manifest.sources]
    if len(source_ids) != len(set(source_ids)):
        raise ManifestError("Grounding source IDs must be unique")

    for source in manifest.sources:
        validate_source_url(source.url)
        if source.rights_profile not in manifest.rights_profiles:
            raise ManifestError(f"Unknown rights profile for {source.id}: {source.rights_profile}")
        if source.verified_content_type != "application/pdf":
            raise ManifestError(f"Grounding source is not a verified PDF: {source.id}")

    return manifest
