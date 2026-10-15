"""Direct clause and statutory reference retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from finance_ca_assistant.domain_index import extract_references


@dataclass(frozen=True)
class ClauseSearchResult:
    """Direct clause lookup result."""

    chunk: Dict[str, Any]
    score: float = 1.0
    method: str = "clause"
    methods: List[str] = field(default_factory=lambda: ["clause"])


class ClauseSearcher:
    """Search chunks by normalized references extracted from the query."""

    def __init__(self, chunks_by_reference: Dict[str, List[Dict[str, Any]]]) -> None:
        self.chunks_by_reference = chunks_by_reference

    @classmethod
    def from_chunks(cls, chunks: List[Dict[str, Any]]) -> "ClauseSearcher":
        mapping: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            for reference in extract_references(str(chunk.get("text") or "")):
                mapping.setdefault(reference.normalized, []).append(dict(chunk))
        return cls(mapping)

    @classmethod
    def from_index(
        cls,
        chunks: List[Dict[str, Any]],
        index: Dict[str, Dict[str, object]],
    ) -> "ClauseSearcher":
        """Load a saved reference index and resolve IDs to current chunks."""

        chunks_by_id = {str(chunk.get("id")): chunk for chunk in chunks}
        mapping: Dict[str, List[Dict[str, Any]]] = {}
        for reference, entry in index.items():
            chunk_ids = entry.get("chunks", []) if isinstance(entry, dict) else []
            mapping[reference] = [
                dict(chunks_by_id[str(chunk_id)])
                for chunk_id in chunk_ids
                if str(chunk_id) in chunks_by_id
            ]
        return cls(mapping)

    def search(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict[str, Any]] = None) -> List[ClauseSearchResult]:
        if top_k <= 0:
            return []
        results: List[ClauseSearchResult] = []
        seen = set()
        for reference in extract_references(query):
            for chunk in self.chunks_by_reference.get(reference.normalized, []):
                if metadata_filter and not all(
                    key in chunk and chunk[key] == value
                    for key, value in metadata_filter.items()
                ):
                    continue
                chunk_id = chunk.get("id")
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                results.append(ClauseSearchResult(chunk=chunk))
        return results[:top_k]
