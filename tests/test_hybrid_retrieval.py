from types import SimpleNamespace

import numpy as np
import pytest

from finance_ca_assistant.pipeline import RAGPipeline
from finance_ca_assistant.retrieval.hybrid_retriever import HybridRetriever
from finance_ca_assistant.retrieval.vector_store_base import InMemoryVectorStore


class RankedResults:
    def __init__(self, ids, method, scale=1):
        self.results = [SimpleNamespace(
            chunk={"id": key, "text": "unrelated" if key == "a" else "query"},
            score=scale / rank, method=method,
        ) for rank, key in enumerate(ids, 1)]

    def search(self, *args, **kwargs):
        return self.results[:kwargs["top_k"]]


def make_retriever(scale=1, reranker=None, **kwargs):
    return HybridRetriever(
        RankedResults(["a", "b", "c", "d", "e", "f"], "dense"),
        RankedResults(["a", "c", "b"], "bm25", scale),
        RankedResults(["a"], "clause"),
        lambda query: np.ones(1), reranker=reranker, **kwargs,
    )


def test_rrf_is_independent_of_raw_score_scale_and_has_no_implicit_reranker():
    first = make_retriever().retrieve("query")
    scaled = make_retriever(1_000_000).retrieve("query")
    assert [(r.chunk["id"], r.score) for r in first] == [(r.chunk["id"], r.score) for r in scaled]
    assert first[0].chunk["id"] == "a"
    assert first[0].score == pytest.approx(1 / 61)
    assert first[0].methods == ["dense", "bm25", "clause"]
    by_id = {result.chunk["id"]: result.score for result in first}
    assert by_id["b"] == pytest.approx(0.45 / 62 + 0.35 / 63)
    assert by_id["c"] == pytest.approx(0.45 / 63 + 0.35 / 62)


def test_dense_filter_precedes_top_k_even_when_excluded_vector_scores_higher():
    store = InMemoryVectorStore()
    store.add(np.array([[1, 0], [0, 1]], dtype=np.float32), [
        {"id": "excluded", "source": "excluded"},
        {"id": "eligible", "source": "allowed"},
    ])
    results = store.search(np.array([1, 0]), top_k=1, metadata_filter={"source": "allowed"})
    assert [result.chunk["id"] for result in results] == ["eligible"]


def test_filters_apply_before_limits_in_every_channel():
    chunks = [
        {"id": str(i), "text": "Form 3CD clause 44 tax reporting", "source": "excluded", "jurisdiction": "IN"}
        for i in range(25)
    ] + [{"id": "eligible", "text": "Form 3CD clause 44 tax reporting", "source": "allowed", "jurisdiction": "IN"}]
    pipeline = RAGPipeline.from_chunks(chunks)
    filt = {"source": "allowed", "jurisdiction": "IN"}
    results = pipeline.retriever.retrieve("Form 3CD clause 44", top_k=1, candidate_k=1, metadata_filter=filt)
    assert [r.chunk["id"] for r in results] == ["eligible"]
    assert results[0].methods == ["dense", "bm25", "clause"]
    assert pipeline.retrieve_chunks("Form 3CD clause 44", metadata_filter={"source": "absent"}) == []
    assert pipeline.answer("Form 3CD clause 44", metadata_filter=filt).retrieved[0].chunk["id"] == "eligible"
    assert pipeline.retrieve_chunks("Form 3CD clause 44", metadata_filter={"missing": None}) == []


def test_explicit_reranker_receives_filtered_candidates_and_can_be_disabled():
    calls = []

    class Reranker:
        def rerank(self, query, results, top_k):
            calls.append(results)
            return results[:top_k]

    pipeline = RAGPipeline.from_chunks([
        {"id": "a", "text": "Section 80C deduction", "source": "allowed"},
        {"id": "b", "text": "Section 80C deduction", "source": "excluded"},
    ], reranker=Reranker())
    for rerank in (True, False):
        pipeline.retriever.retrieve("Section 80C", metadata_filter={"source": "allowed"}, rerank=rerank)
    assert len(calls) == 1
    assert [r.chunk["id"] for r in calls[0]] == ["a"]


@pytest.mark.parametrize("weights", [(-1, 1, 1), (0, 0, 0), (float("nan"), 1, 1), (float("inf"), 1, 1)])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        make_retriever(dense_weight=weights[0], bm25_weight=weights[1], clause_weight=weights[2])


def test_limits_and_zero_weight_channels():
    retriever = make_retriever(dense_weight=0, bm25_weight=0, clause_weight=1)
    assert retriever.retrieve("query", top_k=0) == []
    assert retriever.retrieve("query", top_k=-1) == []
    with pytest.raises(ValueError):
        retriever.retrieve("query", candidate_k=0)
    results = retriever.retrieve("query")
    assert len(results) == 1
    assert results[0].methods == ["clause"]


def test_duplicate_ids_do_not_inflate_channel_contribution():
    retriever = make_retriever()
    merged = {}
    retriever._merge(merged, RankedResults(["a", "a"], "dense").results, 1)
    assert merged["a"].score == pytest.approx(1 / 61)
