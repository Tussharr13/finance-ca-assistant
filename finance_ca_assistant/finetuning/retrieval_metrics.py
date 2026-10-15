"""Score retrieval on held-out (question, chunk) pairs."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

from finance_ca_assistant.finetuning.dataset import QueryPair, page_key

SearchFn = Callable[[str, int], List[Mapping[str, Any]]]


def evaluate_pairs(pairs: Sequence[QueryPair], search: SearchFn, ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """Recall@k and MRR@max(k), at chunk level and at page level.

    Page level counts any chunk from the gold page as a hit, which matches how
    answers cite sources (document and page).
    """

    if not pairs:
        raise ValueError("No evaluation pairs")
    depth = max(ks)
    chunk_hits = {k: 0 for k in ks}
    page_hits = {k: 0 for k in ks}
    chunk_rr = page_rr = 0.0
    for pair in pairs:
        results = search(pair.query, depth)[:depth]
        gold_page = page_key(pair.source, pair.page)
        chunk_rank = next((i for i, c in enumerate(results, 1) if str(c.get("id")) == pair.chunk_id), None)
        page_rank = next(
            (i for i, c in enumerate(results, 1) if page_key(c.get("source"), c.get("page")) == gold_page), None
        )
        for k in ks:
            chunk_hits[k] += bool(chunk_rank and chunk_rank <= k)
            page_hits[k] += bool(page_rank and page_rank <= k)
        chunk_rr += 1 / chunk_rank if chunk_rank else 0.0
        page_rr += 1 / page_rank if page_rank else 0.0
    total = len(pairs)
    metrics = {f"chunk_recall@{k}": chunk_hits[k] / total for k in ks}
    metrics.update({f"page_recall@{k}": page_hits[k] / total for k in ks})
    metrics[f"chunk_mrr@{depth}"] = chunk_rr / total
    metrics[f"page_mrr@{depth}"] = page_rr / total
    metrics["questions"] = float(total)
    return metrics


def dense_search_fn(embedding_provider: Any, chunks: Sequence[Mapping[str, Any]]) -> SearchFn:
    """Dense-only search over ``chunks``; embeds the corpus once."""

    corpus = list(chunks)
    matrix = np.asarray(embedding_provider.embed_texts([str(c.get("text") or "") for c in corpus]), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)

    def search(query: str, k: int) -> List[Mapping[str, Any]]:
        vector = np.asarray(embedding_provider.embed_query(query), dtype=np.float32)
        vector = vector / (np.linalg.norm(vector) or 1.0)
        scores = matrix @ vector
        top = np.argsort(-scores)[:k]
        return [corpus[i] for i in top]

    return search
