"""Grid-search the hybrid RRF weights against the golden retrieval set.

    python3 scripts/sweep_rrf_weights.py --artifacts /path/to/kaggle/artifacts

Only the hybrid channel changes between configurations; dense, BM25 and clause
rankings are fixed, so query embeddings are computed once and reused. Every
configuration is scored with the same benchmark code as scripts/benchmark_retrieval.py,
and the table shows the current defaults (dense 0.45, BM25 0.35, clause 0.20,
rrf_k 60) so the gain or loss is visible at a glance.

This tunes against the same 52 labelled questions the regression gate uses, so
the winner is optimistic. Treat a small gain as unproven until it survives a
held-out split or a second labelled set. Nothing is written to the baseline here;
pass --write-baseline to scripts/benchmark_retrieval.py after you choose weights.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finance_ca_assistant.evaluation.golden_testset import load_golden_testset
from finance_ca_assistant.evaluation.retrieval_benchmark import run_benchmark
from finance_ca_assistant.finetuning.dataset import page_key
from server.pipeline_loader import load_pipeline

DEFAULT_WEIGHTS = {"dense": 0.45, "bm25": 0.35, "clause": 0.20}
DEFAULT_RRF_K = 60
DENSE_GRID = (0.2, 0.35, 0.45, 0.6)
BM25_GRID = (0.2, 0.35, 0.5, 0.7)
CLAUSE_GRID = (0.0, 0.1, 0.2, 0.4)
RRF_K_GRID = (10, 30, 60)
REPORT_METRICS = ("page_mrr@10", "page_recall@1", "page_recall@5", "source_mrr@10", "source_recall@5")


def _grid() -> List[Tuple[float, float, float, int]]:
    combos = itertools.product(DENSE_GRID, BM25_GRID, CLAUSE_GRID, RRF_K_GRID)
    return [combo for combo in combos]


def _cached_query_embedder(embedder):
    """Embed each golden question once; the sweep reuses the same vectors."""
    cache: Dict[str, Any] = {}

    def embed(query: str):
        if query not in cache:
            cache[query] = embedder(query)
        return cache[query]

    return embed


def score_config(pipeline, questions, corpus_sources, corpus_pages, weights, rrf_k) -> Dict[str, float]:
    retriever = pipeline.retriever
    retriever.dense_weight = weights[0]
    retriever.bm25_weight = weights[1]
    retriever.clause_weight = weights[2]
    retriever.rrf_k = rrf_k

    def hybrid(query: str, k: int):
        return [result.chunk for result in retriever.retrieve(query, top_k=k, candidate_k=max(20, k))]

    report = run_benchmark(
        questions,
        {"hybrid": hybrid},
        corpus_sources=corpus_sources,
        corpus_pages=corpus_pages,
    )
    return report["methods"]["hybrid"]["metrics"]


def sweep(pipeline, questions, corpus_sources, corpus_pages, grid=None) -> Dict[str, Any]:
    """Score every configuration; return rows sorted by page MRR@10, best first."""
    retriever = pipeline.retriever
    retriever.query_embedder = _cached_query_embedder(retriever.query_embedder)
    rows = []
    for dense, bm25, clause, rrf_k in grid or _grid():
        metrics = score_config(pipeline, questions, corpus_sources, corpus_pages, (dense, bm25, clause), rrf_k)
        rows.append({
            "weights": {"dense": dense, "bm25": bm25, "clause": clause},
            "rrf_k": rrf_k,
            "metrics": {name: metrics.get(name) for name in REPORT_METRICS},
        })
    rows.sort(key=lambda row: row["metrics"]["page_mrr@10"] or 0.0, reverse=True)
    default = score_config(
        pipeline, questions, corpus_sources, corpus_pages,
        (DEFAULT_WEIGHTS["dense"], DEFAULT_WEIGHTS["bm25"], DEFAULT_WEIGHTS["clause"]), DEFAULT_RRF_K,
    )
    return {
        "default": {"weights": DEFAULT_WEIGHTS, "rrf_k": DEFAULT_RRF_K,
                    "metrics": {name: default.get(name) for name in REPORT_METRICS}},
        "configurations": rows,
    }


def render(result: Dict[str, Any], top: int) -> str:
    def line(row: Dict[str, Any]) -> str:
        w = row["weights"]
        m = row["metrics"]
        cells = " | ".join(f"{m[name]:.3f}" if m[name] is not None else "-" for name in REPORT_METRICS)
        return f"| {w['dense']} | {w['bm25']} | {w['clause']} | {row['rrf_k']} | {cells} |"

    header = "| dense | bm25 | clause | rrf_k | " + " | ".join(REPORT_METRICS) + " |"
    divider = "|" + "---|" * (4 + len(REPORT_METRICS))
    default = {"weights": result["default"]["weights"], "rrf_k": result["default"]["rrf_k"],
               "metrics": result["default"]["metrics"]}
    lines = [
        "# RRF weight sweep",
        "",
        "Current defaults:",
        "",
        header,
        divider,
        line(default),
        "",
        f"Top {top} of {len(result['configurations'])} configurations by page MRR@10:",
        "",
        header,
        divider,
    ]
    lines += [line(row) for row in result["configurations"][:top]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default=os.environ.get("FINANCE_CA_ARTIFACTS_DIR"),
                        help="Folder with chunks.jsonl / embeddings_cache.npz (default: $FINANCE_CA_ARTIFACTS_DIR)")
    parser.add_argument("--golden", default=str(ROOT / "data" / "golden_testset.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "benchmark"))
    parser.add_argument("--top", type=int, default=15, help="Rows to print")
    args = parser.parse_args()
    if not args.artifacts:
        parser.error("--artifacts or FINANCE_CA_ARTIFACTS_DIR is required")

    loaded = load_pipeline(args.artifacts)
    print("Corpus:", loaded.corpus)
    questions = load_golden_testset(args.golden)
    chunks = loaded.pipeline.chunks
    sources = sorted({str(chunk.get("source")) for chunk in chunks})
    pages = sorted({page_key(chunk.get("source"), chunk.get("page")) for chunk in chunks})

    result = sweep(loaded.pipeline, questions, sources, pages)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rrf_sweep.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render(result, args.top)
    (out_dir / "rrf_sweep.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
