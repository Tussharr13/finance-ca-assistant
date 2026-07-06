"""Embedding matrix validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingValidationResult:
    """Result of embedding matrix validation."""

    valid: bool
    reason: str = ""


def validate_embedding_matrix(embeddings: np.ndarray, expected_rows: int) -> EmbeddingValidationResult:
    """Validate shape and numeric health of an embedding matrix."""

    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        return EmbeddingValidationResult(False, "embedding matrix must be 2D")
    if matrix.shape[0] != expected_rows:
        return EmbeddingValidationResult(False, "embedding row count does not match chunk count")
    if not np.isfinite(matrix).all():
        return EmbeddingValidationResult(False, "embedding matrix contains non-finite values")
    return EmbeddingValidationResult(True)

