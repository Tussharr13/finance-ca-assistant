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

    def validate_quoted_claim(
        self,
        claim: str,
        quote: str,
        cited_chunk: Dict[str, Any],
        max_sentences: int = 3,
        min_overlap: float = 0.5,
    ) -> CitationValidationResult:
        """Check a paraphrased claim against a verbatim quote from one chunk.

        The quote must be 1 to ``max_sentences`` complete, consecutive sentences
        of the chunk, so it cannot be cropped to drop "No" or "unless". The
        claim must keep the quote's negation and qualifiers, use only numbers in
        the quote, and draw most of its key words from it. This checks
        attribution, never truth, currency, or applicability.
        """

        source, page = cited_chunk.get("source"), cited_chunk.get("page")
        if not isinstance(source, str) or not source.strip() or not isinstance(page, int) or isinstance(page, bool) or page < 1:
            return CitationValidationResult(False, 0.0, "missing_source_or_page")
        excerpt, chunk_text = _normalize(quote), _normalize(str(cited_chunk.get("text") or ""))
        if not excerpt or not claim.strip():
            return CitationValidationResult(False, 0.0, "empty_claim_or_quote")
        if not _is_sentence_aligned(excerpt, chunk_text, max_sentences):
            return CitationValidationResult(False, 0.0, "quote_not_complete_sentences_from_chunk")
        if _has_negation(claim) != _has_negation(excerpt):
            return CitationValidationResult(False, 0.0, "negation_mismatch")
        if _has_qualifier(excerpt) and not _has_qualifier(claim):
            return CitationValidationResult(False, 0.0, "dropped_qualifier")
        quote_numbers = _numbers(excerpt)
        if any(number not in quote_numbers for number in _numbers(claim)):
            return CitationValidationResult(False, 0.0, "number_not_in_quote")
        claim_tokens = _meaningful_tokens(claim)
        overlap = len(claim_tokens & _meaningful_tokens(excerpt)) / len(claim_tokens) if claim_tokens else 0.0
        if overlap < min_overlap:
            return CitationValidationResult(False, overlap, "insufficient_overlap_with_quote")
        return CitationValidationResult(True, overlap, "quote_verified")

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


_NEGATION = re.compile(r"\b(?:no|not|never|nor|neither|none|nothing|cannot|without|prohibited)\b|n't\b", re.IGNORECASE)
_QUALIFIER = re.compile(r"\b(?:except|unless|provided that|subject to|only if|other than)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
_NUMBER = re.compile(r"(?<![a-zA-Z0-9])\d[\d,]*(?:\.\d+)?(?![a-zA-Z(])")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _has_negation(text: str) -> bool:
    return bool(_NEGATION.search(text))


def _has_qualifier(text: str) -> bool:
    return bool(_QUALIFIER.search(text))


def _numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", "").rstrip(".") for match in _NUMBER.finditer(text)}


def _is_sentence_aligned(quote: str, chunk_text: str, max_sentences: int) -> bool:
    """True if ``quote`` is 1..max_sentences whole consecutive sentences of the chunk.

    The chunk's own start and end count as boundaries, since chunks may begin
    or end mid-sentence.
    """

    starts = {0} | {match.end() for match in _SENTENCE_END.finditer(chunk_text)}
    ends = {len(chunk_text)} | {match.start() for match in _SENTENCE_END.finditer(chunk_text)}
    if len(_SENTENCE_END.findall(quote)) + 1 > max_sentences:
        return False
    index = chunk_text.find(quote)
    while index != -1:
        if index in starts and index + len(quote) in ends:
            return True
        index = chunk_text.find(quote, index + 1)
    return False
