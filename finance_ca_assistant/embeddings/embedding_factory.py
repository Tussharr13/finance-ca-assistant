"""Embedding provider factory and local deterministic embeddings."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np


class EmbeddingProvider(ABC):
    """Interface for all embedding providers."""

    model_name: str
    dimensions: int

    @abstractmethod
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Embed multiple texts into a 2D float32 matrix."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query string."""

        return self.embed_texts([text])[0]


class DeterministicHashEmbeddingProvider(EmbeddingProvider):
    """Free local embedding provider for tests and offline dry-runs.

    It is not semantically strong, but it is deterministic and dependency-free,
    which makes it useful for CI and notebook setup checks.
    """

    def __init__(self, dimensions: int = 384, model_name: str = "hash-local") -> None:
        self.dimensions = dimensions
        self.model_name = model_name

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            tokens = str(text).lower().split()
            for token in tokens or [""]:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        return np.vstack(rows).astype(np.float32)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Hugging Face / sentence-transformers embedding provider.

    The default model is ``BAAI/bge-m3`` because it is strong for RAG retrieval
    and works well with the existing dense + BM25 hybrid pipeline. Imports are
    lazy so tests and offline smoke runs do not need heavyweight ML packages.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        model: Optional[object] = None,
        dimensions: int = 1024,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model = model or self._load_model(model_name=model_name, device=device)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        batch = [" ".join(str(text).split()) or " " for text in texts]
        if not batch:
            return np.empty((0, self.dimensions), dtype=np.float32)

        encoded = self.model.encode(
            batch,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("sentence-transformers encode() must return a 2D embedding matrix")
        self.dimensions = int(matrix.shape[1])
        return matrix

    @staticmethod
    def _load_model(model_name: str, device: Optional[str] = None) -> object:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use Hugging Face embeddings: "
                "pip install sentence-transformers"
            ) from exc

        kwargs = {"device": device} if device else {}
        return SentenceTransformer(model_name, **kwargs)


def create_embedding_provider(name: str = "hash-local", **kwargs: object) -> EmbeddingProvider:
    """Create an embedding provider by name."""

    normalized = name.lower()
    if normalized in {"hash-local", "deterministic", "local"}:
        return DeterministicHashEmbeddingProvider(**kwargs)
    if normalized in {"huggingface", "sentence-transformer", "sentence-transformers", "bge-m3", "baai/bge-m3"}:
        kwargs.setdefault("model_name", "BAAI/bge-m3")
        return SentenceTransformerEmbeddingProvider(**kwargs)
    raise ValueError(f"Unknown embedding provider: {name}")
