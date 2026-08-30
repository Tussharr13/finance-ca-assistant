"""Versioned source registry for authoritative PDF files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional

from finance_ca_assistant.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class SourceRecord:
    """Metadata for one registered source file."""

    source_id: str
    path: str
    version: str
    sha256: str
    size_bytes: int
    registered_at: str
    changed: bool
    title: Optional[str] = None
    url: Optional[str] = None
    authority: Optional[str] = None
    document_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    publication_date: Optional[str] = None
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    financial_year: Optional[str] = None
    assessment_year: Optional[str] = None


class SourceRegistry:
    """Persist source hashes and detect when PDFs changed."""

    def __init__(self, manifest_path: Path | str) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Dict[str, object]]:
        """Load the manifest as a dictionary."""

        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return dict(json.load(handle))

    def save(self, manifest: Dict[str, Dict[str, object]]) -> None:
        """Write the manifest atomically enough for notebook/API use."""

        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def register_source(
        self,
        source_id: str,
        path: str | Path,
        version: str = "latest",
        expected_sha256: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> SourceRecord:
        """Register a file, validate its optional hash, and return change status."""

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        sha256 = compute_sha256(file_path)
        if expected_sha256 and sha256 != expected_sha256:
            raise ValueError(
                f"SHA256 mismatch for {source_id}: expected {expected_sha256}, got {sha256}"
            )

        manifest = self.load()
        previous = manifest.get(source_id)
        changed = previous is None or previous.get("sha256") != sha256
        source_metadata = dict(metadata or {})
        record = SourceRecord(
            source_id=source_id,
            path=str(file_path),
            version=version,
            sha256=sha256,
            size_bytes=file_path.stat().st_size,
            registered_at=datetime.now(timezone.utc).isoformat(),
            changed=changed,
            title=_optional_string(source_metadata.get("title")),
            url=_optional_string(source_metadata.get("url")),
            authority=_optional_string(source_metadata.get("authority")),
            document_type=_optional_string(source_metadata.get("document_type")),
            jurisdiction=_optional_string(source_metadata.get("jurisdiction")),
            publication_date=_optional_string(source_metadata.get("publication_date")),
            effective_from=_optional_string(source_metadata.get("effective_from")),
            effective_to=_optional_string(source_metadata.get("effective_to")),
            financial_year=_optional_string(source_metadata.get("financial_year")),
            assessment_year=_optional_string(source_metadata.get("assessment_year")),
        )
        manifest[source_id] = asdict(record)
        self.save(manifest)
        logger.info("Registered source %s changed=%s", source_id, changed)
        return record


def _optional_string(value: object) -> Optional[str]:
    return None if value is None else str(value)


def compute_sha256(path: str | Path) -> str:
    """Compute SHA256 hash for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
