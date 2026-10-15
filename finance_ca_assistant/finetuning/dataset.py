"""Train/eval splits and hard-negative mining for retrieval fine-tuning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from finance_ca_assistant.retrieval.bm25_index import BM25Index


@dataclass(frozen=True)
class QueryPair:
    query: str
    chunk_id: str
    source: str
    page: Any


def page_key(source: Any, page: Any) -> str:
    return f"{source}#{page}"


def split_pairs(
    records: Sequence[Mapping[str, Any]],
    eval_fraction: float = 0.15,
    seed: str = "v1",
) -> Tuple[List[QueryPair], List[QueryPair]]:
    """Split question records into train and eval by source page.

    All questions from one page land on the same side, so the eval set measures
    retrieval for pages the model never trained on. The split is a stable hash,
    so reruns and added records never move a page between sides.
    """

    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    train: List[QueryPair] = []
    evaluation: List[QueryPair] = []
    for record in records:
        key = page_key(record.get("source"), record.get("page"))
        bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        target = evaluation if bucket < eval_fraction else train
        for question in record.get("questions") or []:
            target.append(QueryPair(question, str(record["chunk_id"]), str(record.get("source")), record.get("page")))
    return train, evaluation


def mine_hard_negatives(
    pairs: Sequence[QueryPair],
    chunks: Sequence[Mapping[str, Any]],
    skip_top: int = 2,
    search_depth: int = 15,
) -> List[Dict[str, str]]:
    """Build (anchor, positive, negative) triplets using BM25 near-misses.

    The negative is a keyword-similar chunk from a different page. The first
    ``skip_top`` candidates are skipped because the very best keyword matches
    are the likeliest to be unlabeled true answers.
    """

    by_id = {str(chunk.get("id")): chunk for chunk in chunks}
    index = BM25Index()
    index.add([dict(chunk) for chunk in chunks])
    triplets: List[Dict[str, str]] = []
    for pair in pairs:
        positive = by_id.get(pair.chunk_id)
        if positive is None:
            continue
        own_page = page_key(pair.source, pair.page)
        candidates = [
            result.chunk
            for result in index.search(pair.query, top_k=search_depth)
            if page_key(result.chunk.get("source"), result.chunk.get("page")) != own_page
        ]
        if not candidates:
            continue
        negative = candidates[min(skip_top, len(candidates) - 1)]
        triplets.append(
            {"anchor": pair.query, "positive": str(positive.get("text") or ""), "negative": str(negative.get("text") or "")}
        )
    return triplets
