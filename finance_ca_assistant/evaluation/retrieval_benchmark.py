"""Benchmark retrieval per method on labelled golden questions, and gate regressions.

Each question is run through hybrid retrieval and through each method alone
(dense, BM25, clause), so a regression can be traced to the method that caused
it. Latency is reported but never gated: it depends on the machine.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from finance_ca_assistant.domain_index import extract_references
from finance_ca_assistant.evaluation.golden_testset import GoldenQuestion
from finance_ca_assistant.finetuning.dataset import page_key

SearchFn = Callable[[str, int], List[Mapping[str, Any]]]

DEFAULT_KS = (1, 5, 10)
# Hybrid metrics compared against the baseline; all are higher-is-better.
GATED_PREFIXES = ("source_recall@", "source_mrr@", "page_recall@", "page_mrr@", "section_recall@")


def method_search_fns(pipeline: Any) -> Dict[str, SearchFn]:
    """Search functions for hybrid retrieval and each of its methods alone."""

    retriever = pipeline.retriever

    def hybrid(query: str, k: int) -> List[Mapping[str, Any]]:
        return [result.chunk for result in retriever.retrieve(query, top_k=k, candidate_k=max(20, k))]

    def dense(query: str, k: int) -> List[Mapping[str, Any]]:
        return [r.chunk for r in retriever.vector_store.search(retriever.query_embedder(query), top_k=k)]

    def bm25(query: str, k: int) -> List[Mapping[str, Any]]:
        return [r.chunk for r in retriever.bm25_index.search(query, top_k=k)]

    def clause(query: str, k: int) -> List[Mapping[str, Any]]:
        return [r.chunk for r in retriever.clause_searcher.search(query, top_k=k)]

    return {"hybrid": hybrid, "dense": dense, "bm25": bm25, "clause": clause}


def run_benchmark(
    questions: Sequence[GoldenQuestion],
    search_fns: Mapping[str, SearchFn],
    corpus_sources: Optional[Sequence[str]] = None,
    ks: Sequence[int] = DEFAULT_KS,
    corpus_pages: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Score every method on every question that the corpus can answer.

    A question whose expected sources are all missing from ``corpus_sources``,
    or whose expected pages are all missing from ``corpus_pages`` (page keys
    from :func:`page_key`), is skipped and listed: it is a corpus or labelling
    gap, not a retrieval miss. A bounded build that keeps only the first pages
    of each PDF would otherwise read as a retrieval failure.
    """

    depth = max(ks)
    available = set(corpus_sources) if corpus_sources is not None else None
    available_pages = set(corpus_pages) if corpus_pages is not None else None
    scored: List[GoldenQuestion] = []
    skipped: List[Dict[str, str]] = []
    for question in questions:
        if available is not None and not available & set(question.correct_sources):
            skipped.append({"id": question.id, "reason": "expected source not in corpus"})
        elif available_pages is not None and question.expected_pages and not available_pages & set(question.expected_pages):
            skipped.append({"id": question.id, "reason": "expected pages not in corpus"})
        else:
            scored.append(question)

    methods: Dict[str, Any] = {}
    misses: List[Dict[str, Any]] = []
    per_question_hybrid: List[Dict[str, Any]] = []
    for name, search in search_fns.items():
        rows = []
        latencies = []
        for question in scored:
            start = time.perf_counter()
            results = list(search(question.question, depth))[:depth]
            latencies.append((time.perf_counter() - start) * 1000)
            row = _score_question(question, results, ks)
            rows.append(row)
            if row["source_rank"] is None:
                misses.append({
                    "id": question.id,
                    "method": name,
                    "retrieved": [f"{c.get('source')}#{c.get('page')}" for c in results[:3]],
                })
        methods[name] = {"metrics": _aggregate(rows, ks), "latency_ms": _latency(latencies)}
        if name == "hybrid":
            per_question_hybrid = rows

    by_category: Dict[str, Dict[str, float]] = {}
    for category in sorted({q.category for q in scored}):
        rows = [row for q, row in zip(scored, per_question_hybrid) if q.category == category]
        if rows:
            by_category[category] = _aggregate(rows, ks)

    return {
        "golden_ids": sorted(q.id for q in questions),
        "questions_scored": len(scored),
        "skipped": skipped,
        "methods": methods,
        "hybrid_by_category": by_category,
        "misses": misses,
    }


def check_gate(report: Mapping[str, Any], baseline: Mapping[str, Any], tolerance: float = 0.02) -> List[str]:
    """Return one message per hybrid metric that fell more than ``tolerance`` below the baseline."""

    if report.get("golden_ids") != baseline.get("golden_ids"):
        return ["Golden set differs from the baseline's; review the new results and write a new baseline."]
    current = report["methods"]["hybrid"]["metrics"]
    previous = baseline["methods"]["hybrid"]["metrics"]
    failures = []
    for name, old in sorted(previous.items()):
        if not name.startswith(GATED_PREFIXES) or name not in current:
            continue
        if current[name] < old - tolerance:
            failures.append(f"{name} fell from {old:.3f} to {current[name]:.3f}")
    return failures


def render_markdown(report: Mapping[str, Any], gate_failures: Optional[Sequence[str]] = None) -> str:
    """Human-readable report: per-method table, hybrid by category, misses."""

    lines = [
        "# Retrieval benchmark",
        "",
        f"Questions scored: {report['questions_scored']} of {len(report['golden_ids'])}",
        "",
    ]
    methods = report["methods"]
    if methods:
        names = list(next(iter(methods.values()))["metrics"])
        lines.append("| Method | " + " | ".join(names) + " | p50 ms | p95 ms |")
        lines.append("|---" * (len(names) + 3) + "|")
        for method, data in methods.items():
            values = " | ".join(_fmt(data["metrics"][n]) for n in names)
            latency = data["latency_ms"]
            lines.append(f"| {method} | {values} | {latency['p50']:.1f} | {latency['p95']:.1f} |")
    if report["hybrid_by_category"]:
        lines.extend(["", "## Hybrid by category", ""])
        for category, metrics in report["hybrid_by_category"].items():
            summary = ", ".join(f"{n} {_fmt(v)}" for n, v in metrics.items())
            lines.append(f"- **{category}**: {summary}")
    if gate_failures is not None:
        lines.extend(["", "## Regression gate", ""])
        lines.extend(f"- FAIL: {failure}" for failure in gate_failures)
        if not gate_failures:
            lines.append("- Passed")
    lines.extend(["", "## Source misses (top 3 retrieved)", ""])
    lines.extend(f"- {m['id']} [{m['method']}]: {', '.join(m['retrieved']) or 'nothing'}" for m in report["misses"])
    if not report["misses"]:
        lines.append("- None")
    if report["skipped"]:
        lines.extend(["", "## Skipped", ""])
        lines.extend(f"- {s['id']}: {s['reason']}" for s in report["skipped"])
    return "\n".join(lines) + "\n"


def _score_question(question: GoldenQuestion, results: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> Dict[str, Any]:
    sources = set(question.correct_sources)
    pages = set(question.expected_pages)
    source_rank = next((i for i, c in enumerate(results, 1) if c.get("source") in sources), None)
    page_rank = None
    if pages:
        page_rank = next((i for i, c in enumerate(results, 1) if page_key(c.get("source"), c.get("page")) in pages), None)
    section_recall = {}
    if question.expected_sections:
        expected = set(question.expected_sections)
        for k in ks:
            found = set()
            for chunk in results[:k]:
                found.update(_references(chunk))
            section_recall[k] = len(found & expected) / len(expected)
    return {"source_rank": source_rank, "page_rank": page_rank, "has_pages": bool(pages), "section_recall": section_recall}


def _references(chunk: Mapping[str, Any]) -> List[str]:
    if "references" in chunk:
        return list(chunk["references"])
    return [reference.normalized for reference in extract_references(str(chunk.get("text") or ""))]


def _aggregate(rows: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> Dict[str, float]:
    depth = max(ks)
    metrics: Dict[str, float] = {}
    if not rows:
        return metrics
    for k in ks:
        metrics[f"source_recall@{k}"] = _share(rows, "source_rank", k)
    metrics[f"source_mrr@{depth}"] = _mrr(rows, "source_rank")
    page_rows = [row for row in rows if row["has_pages"]]
    if page_rows:
        for k in ks:
            metrics[f"page_recall@{k}"] = _share(page_rows, "page_rank", k)
        metrics[f"page_mrr@{depth}"] = _mrr(page_rows, "page_rank")
    section_rows = [row for row in rows if row["section_recall"]]
    if section_rows:
        for k in ks:
            metrics[f"section_recall@{k}"] = sum(row["section_recall"][k] for row in section_rows) / len(section_rows)
    return metrics


def _share(rows: Sequence[Mapping[str, Any]], key: str, k: int) -> float:
    return sum(1 for row in rows if row[key] is not None and row[key] <= k) / len(rows)


def _mrr(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(1 / row[key] for row in rows if row[key]) / len(rows)


def _latency(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {"p50": statistics.median(ordered), "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]}


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)
