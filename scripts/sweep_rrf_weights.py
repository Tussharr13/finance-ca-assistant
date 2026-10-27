"""Grid-search the hybrid RRF weights against the golden retrieval set.

    python3 scripts/sweep_rrf_weights.py --artifacts /path/to/kaggle/artifacts
    python3 scripts/sweep_rrf_weights.py --artifacts ... --held-out   # two-fold check

--held-out tunes on one half of the golden set, scores that choice on the other
half, and compares it with the current defaults. Writes rrf_holdout.{md,json}.

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


def split_halves(questions) -> Tuple[List[Any], List[Any]]:
    """Two folds by sorted question ID, so the split is stable across runs."""
    ordered = sorted(questions, key=lambda question: question.id)
    return ordered[0::2], ordered[1::2]


def _best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Highest page MRR@10; ties go to higher page recall@1, then the earlier row."""
    return max(rows, key=lambda row: (row["metrics"]["page_mrr@10"] or 0.0, row["metrics"]["page_recall@1"] or 0.0))


def held_out(pipeline, questions, corpus_sources, corpus_pages, grid=None) -> Dict[str, Any]:
    """Two-fold check: choose weights on one half, score them on the other half.

    Each fold's chosen configuration is compared with the current defaults on
    questions it never saw. A gain only counts if it holds on both folds.
    """
    folds = []
    for tune_half, test_half in (split_halves(questions), split_halves(questions)[::-1]):
        tuned = sweep(pipeline, tune_half, corpus_sources, corpus_pages, grid=grid)
        chosen = _best(tuned["configurations"])
        w = chosen["weights"]
        chosen_test = score_config(pipeline, test_half, corpus_sources, corpus_pages,
                                   (w["dense"], w["bm25"], w["clause"]), chosen["rrf_k"])
        default_test = score_config(pipeline, test_half, corpus_sources, corpus_pages,
                                    (DEFAULT_WEIGHTS["dense"], DEFAULT_WEIGHTS["bm25"], DEFAULT_WEIGHTS["clause"]),
                                    DEFAULT_RRF_K)
        folds.append({
            "tuned_on": len(tune_half),
            "tested_on": len(test_half),
            "chosen": {"weights": w, "rrf_k": chosen["rrf_k"], "tune_metrics": chosen["metrics"]},
            "held_out": {
                "chosen": {name: chosen_test.get(name) for name in REPORT_METRICS},
                "default": {name: default_test.get(name) for name in REPORT_METRICS},
            },
        })
    return {"folds": folds}


def render_held_out(result: Dict[str, Any]) -> str:
    lines = ["# Held-out RRF check", "",
             "Weights chosen on one half of the golden set, scored on the other half (two folds).", ""]
    for index, fold in enumerate(result["folds"], start=1):
        w = fold["chosen"]["weights"]
        lines += [
            f"## Fold {index}: tuned on {fold['tuned_on']}, tested on {fold['tested_on']}",
            "",
            f"Chosen: dense {w['dense']}, BM25 {w['bm25']}, clause {w['clause']}, rrf_k {fold['chosen']['rrf_k']}",
            "",
            "| set | " + " | ".join(REPORT_METRICS) + " |",
            "|" + "---|" * (1 + len(REPORT_METRICS)),
        ]
        for label, metrics in (("held-out, chosen", fold["held_out"]["chosen"]),
                               ("held-out, default", fold["held_out"]["default"])):
            cells = " | ".join(f"{metrics[name]:.3f}" if metrics[name] is not None else "-" for name in REPORT_METRICS)
            lines.append(f"| {label} | {cells} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held-out", action="store_true",
                        help="Run the two-fold held-out check instead of the full sweep table")
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.held_out:
        held = held_out(loaded.pipeline, questions, sources, pages)
        (out_dir / "rrf_holdout.json").write_text(json.dumps(held, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        markdown = render_held_out(held)
        (out_dir / "rrf_holdout.md").write_text(markdown, encoding="utf-8")
        print(markdown)
        return 0

    result = sweep(loaded.pipeline, questions, sources, pages)
    (out_dir / "rrf_sweep.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render(result, args.top)
    (out_dir / "rrf_sweep.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
