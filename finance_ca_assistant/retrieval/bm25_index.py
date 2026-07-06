"""Lightweight BM25 keyword retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class BM25Result:
    """BM25 search result."""

    chunk: Dict[str, Any]
    score: float
    method: str = "bm25"
    methods: List[str] = field(default_factory=lambda: ["bm25"])


class BM25Index:
    """Small BM25 index for exact CA terms and clause wording."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.doc_terms: List[Counter[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_doc_len = 0.0

    def add(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = [dict(chunk) for chunk in chunks]
        self.doc_terms = []
        self.doc_freq = Counter()
        lengths: List[int] = []
        for chunk in self.chunks:
            terms = Counter(tokenize(str(chunk.get("text") or "")))
            self.doc_terms.append(terms)
            lengths.append(sum(terms.values()))
            for term in terms:
                self.doc_freq[term] += 1
        self.avg_doc_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    def search(self, query: str, top_k: int) -> List[BM25Result]:
        query_terms = tokenize(query)
        scores = []
        total_docs = len(self.chunks)
        for index, terms in enumerate(self.doc_terms):
            doc_len = sum(terms.values()) or 1
            score = 0.0
            for term in query_terms:
                if term not in terms:
                    continue
                idf = math.log(1 + (total_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                numerator = terms[term] * (self.k1 + 1)
                denominator = terms[term] + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1))
                score += idf * numerator / denominator
            if score > 0:
                scores.append(BM25Result(chunk=self.chunks[index], score=score))
        return sorted(scores, key=lambda result: result.score, reverse=True)[:top_k]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:\([a-zA-Z0-9]+\))*", text.lower())

