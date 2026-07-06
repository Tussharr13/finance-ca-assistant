"""Reranker selection and reranking implementations."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import List, Optional

from finance_ca_assistant.retrieval.bm25_index import tokenize


class KeywordOverlapReranker:
    """Simple local reranker based on query/chunk token overlap."""

    def rerank(self, query: str, results: List[object], top_k: int) -> List[object]:
        query_terms = set(tokenize(query))

        def score(result: object) -> float:
            chunk = getattr(result, "chunk")
            text_terms = set(tokenize(str(chunk.get("text") or "")))
            overlap = len(query_terms & text_terms) / (len(query_terms) or 1)
            return float(getattr(result, "score", 0.0)) + overlap

        return sorted(results, key=score, reverse=True)[:top_k]


class CrossEncoderReranker:
    """Hugging Face cross-encoder reranker.

    The default model, ``BAAI/bge-reranker-v2-m3``, scores query-passage pairs
    directly and is intended to run after hybrid retrieval has produced a
    candidate set. The model import is lazy so offline tests and smoke runs keep
    working without heavyweight ML dependencies.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model: Optional[object] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.model = model or self._load_model(model_name=model_name, device=device)

    def rerank(self, query: str, results: List[object], top_k: int) -> List[object]:
        """Return top results sorted by cross-encoder relevance."""

        if not results:
            return []

        pairs = [(query, str(getattr(result, "chunk", {}).get("text") or "")) for result in results]
        scores = self._score_pairs(pairs)
        scored_results = [
            self._with_rerank_metadata(result=result, score=float(score))
            for result, score in zip(results, scores)
        ]
        return sorted(scored_results, key=lambda result: float(getattr(result, "score", 0.0)), reverse=True)[:top_k]

    def _score_pairs(self, pairs: List[tuple[str, str]]) -> List[float]:
        if hasattr(self.model, "predict"):
            raw_scores = self.model.predict(pairs)
        elif hasattr(self.model, "compute_score"):
            raw_scores = self.model.compute_score(pairs)
        else:
            raise TypeError("Reranker model must expose predict() or compute_score()")

        if isinstance(raw_scores, (int, float)):
            return [float(raw_scores)]
        return [float(score) for score in raw_scores]

    @staticmethod
    def _with_rerank_metadata(result: object, score: float) -> object:
        methods = list(getattr(result, "methods", []) or [])
        if "rerank" not in methods:
            methods.append("rerank")
        if is_dataclass(result):
            return replace(result, score=score, methods=methods)
        setattr(result, "score", score)
        setattr(result, "methods", methods)
        return result

    @staticmethod
    def _load_model(model_name: str, device: Optional[str] = None) -> object:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use Hugging Face reranking: "
                "pip install sentence-transformers"
            ) from exc

        kwargs = {"device": device} if device else {}
        return CrossEncoder(model_name, **kwargs)


def create_reranker(name: str = "keyword", **kwargs: object) -> object:
    """Create a reranker by name."""

    normalized = name.lower()
    if normalized in {"keyword", "keyword-overlap", "local"}:
        return KeywordOverlapReranker()
    if normalized in {
        "huggingface",
        "cross-encoder",
        "bge-reranker",
        "bge-reranker-v2-m3",
        "baai/bge-reranker-v2-m3",
    }:
        kwargs.setdefault("model_name", "BAAI/bge-reranker-v2-m3")
        return CrossEncoderReranker(**kwargs)
    raise ValueError(f"Unknown reranker provider: {name}")


def select_reranker(candidate_count: int, high_latency_constraint: bool = False) -> Optional[KeywordOverlapReranker]:
    """Select a reranker according to the brief's conditional logic."""

    if candidate_count <= 5:
        return None
    return KeywordOverlapReranker()
