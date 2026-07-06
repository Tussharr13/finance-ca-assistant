"""Hybrid retrieval across dense, BM25, and direct clause search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from finance_ca_assistant.retrieval.bm25_index import BM25Index
from finance_ca_assistant.retrieval.clause_searcher import ClauseSearcher
from finance_ca_assistant.retrieval.reranker_factory import select_reranker
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
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.clause_searcher = clause_searcher
        self.query_embedder = query_embedder
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.clause_weight = clause_weight
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> List[HybridSearchResult]:
        dense = self.vector_store.search(
            self.query_embedder(query),
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )
        sparse = self.bm25_index.search(query, top_k=candidate_k)
        clause = self.clause_searcher.search(query, top_k=candidate_k)

        merged: Dict[str, HybridSearchResult] = {}
        self._merge(merged, dense, self.dense_weight)
        self._merge(merged, sparse, self.bm25_weight)
        self._merge(merged, clause, self.clause_weight)
        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)

        selected_reranker = None
        if rerank:
            selected_reranker = self.reranker or select_reranker(len(results))
        if selected_reranker:
            results = selected_reranker.rerank(query, results, top_k=top_k)
        return results[:top_k]

    def _merge(self, merged: Dict[str, HybridSearchResult], results: List[object], weight: float) -> None:
        for result in results:
            chunk = getattr(result, "chunk")
            chunk_id = str(chunk.get("id") or id(chunk))
            method = getattr(result, "method", "unknown")
            score = float(getattr(result, "score", 0.0)) * weight
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
