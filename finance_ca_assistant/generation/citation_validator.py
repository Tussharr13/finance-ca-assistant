"""Validate whether cited chunks support answer claims."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CitationValidationResult:
    """Result for one claim/chunk support check."""

    valid: bool
    overlap_score: float
    reason: str = ""


class CitationValidator:
    """Validate claim support using conservative token overlap."""

    def __init__(self, min_overlap: float = 0.6) -> None:
        self.min_overlap = min_overlap

    def validate_claim(self, claim: str, cited_chunk: Dict[str, Any]) -> CitationValidationResult:
        claim_tokens = _meaningful_tokens(claim)
        source_tokens = _meaningful_tokens(str(cited_chunk.get("text") or ""))
        if not claim_tokens:
            return CitationValidationResult(False, 0.0, "empty_claim")
        overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
        return CitationValidationResult(
            valid=overlap >= self.min_overlap,
            overlap_score=overlap,
            reason="" if overlap >= self.min_overlap else "claim_not_supported_by_cited_chunk",
        )


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "and",
        "or",
        "is",
        "are",
        "this",
        "that",
        "for",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }

