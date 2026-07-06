"""Vector store interfaces and in-memory implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class VectorSearchResult:
    """Dense vector search result."""

    chunk: Dict[str, Any]
    score: float
    method: str = "dense"
    methods: List[str] = field(default_factory=lambda: ["dense"])


class VectorStoreBase(ABC):
    """Abstract vector store backend."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """Add vectors and chunk metadata."""

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search nearest chunks."""


class InMemoryVectorStore(VectorStoreBase):
    """Dependency-free vector store for Kaggle and tests."""

    def __init__(self) -> None:
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")
        if matrix.shape[0] != len(chunks):
            raise ValueError("chunk count must match embedding rows")
        self.embeddings = matrix if self.embeddings is None else np.vstack([self.embeddings, matrix])
        self.chunks.extend(dict(chunk) for chunk in chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        if self.embeddings is None or not self.chunks:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        indexes = [
            index
            for index, chunk in enumerate(self.chunks)
            if not metadata_filter or all(chunk.get(key) == value for key, value in metadata_filter.items())
        ]
        if not indexes:
            return []
        scores = cosine_similarity(query, self.embeddings[indexes])
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            VectorSearchResult(chunk=self.chunks[indexes[local]], score=float(scores[local]))
            for local in ranked
        ]


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominator = np.where(matrix_norms * query_norm == 0, 1e-12, matrix_norms * query_norm)
    return np.dot(matrix, query) / denominator

