"""Benchmark retrieval on the golden set and gate regressions against a baseline.

    python3 scripts/benchmark_retrieval.py --artifacts /path/to/kaggle/artifacts
    python3 scripts/benchmark_retrieval.py --artifacts ... --write-baseline

Exits 1 when a hybrid quality metric falls more than --tolerance below the
baseline, or when the golden set no longer matches the baseline's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finance_ca_assistant.evaluation.golden_testset import load_golden_testset
from finance_ca_assistant.finetuning.dataset import page_key
from finance_ca_assistant.evaluation.retrieval_benchmark import (
    check_gate,
    method_search_fns,
    render_markdown,
    run_benchmark,
)
from server.pipeline_loader import load_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default=os.environ.get("FINANCE_CA_ARTIFACTS_DIR"),
                        help="Folder with chunks.jsonl / embeddings_cache.npz (default: $FINANCE_CA_ARTIFACTS_DIR)")
    parser.add_argument("--golden", default=str(ROOT / "data" / "golden_testset.json"))
    parser.add_argument("--baseline", default=str(ROOT / "data" / "retrieval_baseline.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "benchmark"))
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--write-baseline", action="store_true", help="Save this run as the new baseline")
    args = parser.parse_args()

    if not args.artifacts:
        parser.error("--artifacts or FINANCE_CA_ARTIFACTS_DIR is required; an empty corpus has nothing to benchmark")

    loaded = load_pipeline(args.artifacts)
    print("Corpus:", loaded.corpus)
    questions = load_golden_testset(args.golden)
    chunks = loaded.pipeline.chunks
    sources = sorted({str(chunk.get("source")) for chunk in chunks})
    pages = sorted({page_key(chunk.get("source"), chunk.get("page")) for chunk in chunks})
    report = run_benchmark(questions, method_search_fns(loaded.pipeline), corpus_sources=sources, corpus_pages=pages)
    # Keep the baseline portable: describe the corpus without the local path.
    report["corpus"] = {
        "chunks": len(chunks),
        "sources": sources,
        "pages": len(pages),
        "detail": loaded.corpus.replace(f" from {Path(args.artifacts)}", ""),
    }

    baseline_path = Path(args.baseline)
    gate_failures = None
    if args.write_baseline:
        baseline_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Baseline written:", baseline_path)
    elif baseline_path.exists():
        gate_failures = check_gate(report, json.loads(baseline_path.read_text(encoding="utf-8")), args.tolerance)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(report, gate_failures)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    if gate_failures is None and not args.write_baseline:
        print(f"No baseline at {baseline_path}; run with --write-baseline to create one.")
    return 1 if gate_failures else 0


if __name__ == "__main__":
    sys.exit(main())
