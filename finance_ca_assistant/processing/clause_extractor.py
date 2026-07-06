"""Build direct CA-domain clause and section indexes from chunks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List

from finance_ca_assistant.domain_index import extract_references


class ClauseExtractor:
    """Extract normalized references and map them to chunk/page/source metadata."""

    def build_index(self, chunks: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, object]]:
        index: DefaultDict[str, Dict[str, object]] = defaultdict(
            lambda: {"chunks": [], "pages": [], "sources": []}
        )
        for fallback_index, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("id") or f"chunk-{fallback_index}")
            page = int(chunk.get("page") or 0)
            source = str(chunk.get("source") or "")
            for reference in extract_references(str(chunk.get("text") or "")):
                entry = index[reference.normalized]
                _append_unique(entry["chunks"], chunk_id)
                if page:
                    _append_unique(entry["pages"], page)
                if source:
                    _append_unique(entry["sources"], source)
        return dict(index)


def _append_unique(values: List[object], value: object) -> None:
    if value not in values:
        values.append(value)

