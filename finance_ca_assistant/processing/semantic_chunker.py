"""Clause-aware semantic chunking for CA-domain documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class TextChunk:
    """A text chunk with metadata suitable for retrieval."""

    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


class SemanticChunker:
    """Chunk CA documents while respecting clauses, rules, and paragraphs."""

    def __init__(self, max_chars: int = 1200, overlap: int = 150) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap < 0 or overlap >= max_chars:
            raise ValueError("overlap must be non-negative and smaller than max_chars")
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, text: str, document_type: str = "generic") -> List[TextChunk]:
        """Split text into metadata-bearing chunks."""

        blocks = self._structured_blocks(text, document_type)
        chunks: List[TextChunk] = []
        for block in blocks:
            for part in self._split_large_block(block):
                chunks.append(TextChunk(text=part, metadata={"document_type": document_type}))
        return chunks

    def _structured_blocks(self, text: str, document_type: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if document_type == "form_3cd":
            return _split_by_heading(text, r"(?=^\s*Clause\s+\d+[A-Za-z]?\b)")
        if document_type in {"gst_rules", "tax_act"}:
            return _split_by_heading(text, r"(?=^\s*(?:Section|Rule)\s+\d+[A-Za-z]?\b)")
        if document_type == "accounting_standard":
            return _split_by_heading(text, r"(?=^\s*\d+\.\s+)")
        return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]

    def _split_large_block(self, block: str) -> Iterable[str]:
        if len(block) <= self.max_chars:
            yield block.strip()
            return
        start = 0
        while start < len(block):
            hard_end = min(start + self.max_chars, len(block))
            # The last window must consume the remainder. Choosing a boundary
            # inside it and then applying overlap can otherwise return to the
            # same ``start`` forever when only a trailing character remains.
            end = (
                len(block)
                if hard_end >= len(block)
                else _choose_boundary(block, start, hard_end)
            )
            part = block[start:end].strip()
            if part:
                yield part
            if end >= len(block):
                break
            next_start = max(0, end - self.overlap)
            # Be defensive if a future boundary strategy returns a split too
            # close to the current start to accommodate the configured overlap.
            start = next_start if next_start > start else end


def _split_by_heading(text: str, pattern: str) -> List[str]:
    parts = [part.strip() for part in re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)]
    return [part for part in parts if part]


def _choose_boundary(text: str, start: int, hard_end: int) -> int:
    minimum = start + int((hard_end - start) * 0.65)
    best_end = -1
    for marker in ["\n\n", ". ", "; ", "\n"]:
        candidate = text.rfind(marker, minimum, hard_end)
        if candidate >= 0:
            candidate_end = candidate + len(marker)
            if candidate_end > best_end:
                best_end = candidate_end
    return best_end if best_end > minimum else hard_end
