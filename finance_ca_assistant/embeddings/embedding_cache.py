"""Compressed embedding cache for notebooks and API startup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class CachedEmbeddings:
    """Embeddings loaded from cache."""

    embeddings: np.ndarray
    metadata: List[Dict[str, Any]]
    model_name: str


class EmbeddingCache:
    """Persist embeddings and chunk metadata in a compressed NPZ file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], model_name: str) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("metadata count must match embedding rows")
        np.savez_compressed(
            self.path,
            embeddings=vectors,
            metadata_json=json.dumps(metadata),
            model_name=model_name,
        )

    def load(self) -> CachedEmbeddings:
        if not self.path.exists():
            raise FileNotFoundError(f"Embedding cache not found: {self.path}")
        data = np.load(self.path, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))
        return CachedEmbeddings(
            embeddings=np.asarray(data["embeddings"], dtype=np.float32),
            metadata=metadata,
            model_name=str(data["model_name"]),
        )

