"""Chunk quality validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class ChunkQualityReport:
    """Quality status for a set of chunks."""

    valid: bool
    issues: List[Dict[str, object]] = field(default_factory=list)


class ChunkQualityValidator:
    """Validate chunk fields and minimum text quality."""

    def __init__(self, min_chars: int = 80) -> None:
        self.min_chars = min_chars

    def validate(self, chunks: Iterable[Dict[str, Any]]) -> ChunkQualityReport:
        issues: List[Dict[str, object]] = []
        for index, chunk in enumerate(chunks):
            text = str(chunk.get("text") or "")
            if len(text.strip()) < self.min_chars:
                issues.append({"index": index, "id": chunk.get("id"), "reason": "too_short"})
            if not chunk.get("source"):
                issues.append({"index": index, "id": chunk.get("id"), "reason": "missing_source"})
            if not chunk.get("page"):
                issues.append({"index": index, "id": chunk.get("id"), "reason": "missing_page"})
        return ChunkQualityReport(valid=not issues, issues=issues)

