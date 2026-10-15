"""Hybrid retrieval across dense, BM25, and direct clause search."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from finance_ca_assistant.retrieval.bm25_index import BM25Index
from finance_ca_assistant.retrieval.clause_searcher import ClauseSearcher
from finance_ca_assistant.retrieval.vector_store_base import VectorStoreBase


@dataclass(frozen=True)
class HybridSearchResult:
    """Merged retrieval result with method provenance."""

    chunk: Dict[str, Any]
    score: float
    methods: List[str] = field(default_factory=list)


class HybridRetriever:
    """Fuse dense vector, sparse keyword, and direct clause results."""

    def __init__(
        self,
        vector_store: VectorStoreBase,
        bm25_index: BM25Index,
        clause_searcher: ClauseSearcher,
        query_embedder: Callable[[str], np.ndarray],
        dense_weight: float = 0.45,
        bm25_weight: float = 0.35,
        clause_weight: float = 0.20,
        reranker: Optional[object] = None,
        rrf_k: int = 60,
    ) -> None:
        weights = (dense_weight, bm25_weight, clause_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(weights):
            raise ValueError("Retrieval weights must be finite, non-negative, and not all zero")
        if not isinstance(rrf_k, int) or rrf_k < 0:
            raise ValueError("rrf_k must be a non-negative integer")
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.clause_searcher = clause_searcher
        self.query_embedder = query_embedder
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.clause_weight = clause_weight
        self.reranker = reranker
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> List[HybridSearchResult]:
        """Fuse ranks using weighted RRF; filters apply before candidate limits.

        Scores express ranking agreement, not confidence or probability.
        Reranking only runs when a reranker was explicitly configured.
        """
        if top_k <= 0:
            return []
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        candidate_k = max(candidate_k, top_k)
        dense = self.vector_store.search(
            self.query_embedder(query),
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )
        sparse = self.bm25_index.search(query, top_k=candidate_k, metadata_filter=metadata_filter)
        clause = self.clause_searcher.search(query, top_k=candidate_k, metadata_filter=metadata_filter)

        merged: Dict[str, HybridSearchResult] = {}
        self._merge(merged, dense, self.dense_weight)
        self._merge(merged, sparse, self.bm25_weight)
        self._merge(merged, clause, self.clause_weight)
        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)

        selected_reranker = self.reranker if rerank else None
        if selected_reranker:
            results = selected_reranker.rerank(query, results, top_k=top_k)
        return results[:top_k]

    def _merge(self, merged: Dict[str, HybridSearchResult], results: List[object], weight: float) -> None:
        if weight == 0:
            return
        seen = set()
        for rank, result in enumerate(results, start=1):
            chunk = getattr(result, "chunk")
            chunk_id = str(chunk.get("id") or id(chunk))
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            method = getattr(result, "method", "unknown")
            score = weight / (self.rrf_k + rank)
            if chunk_id not in merged:
                merged[chunk_id] = HybridSearchResult(chunk=chunk, score=score, methods=[method])
            else:
                existing = merged[chunk_id]
                methods = existing.methods + ([] if method in existing.methods else [method])
                merged[chunk_id] = HybridSearchResult(
                    chunk=existing.chunk,
                    score=existing.score + score,
                    methods=methods,
                )
