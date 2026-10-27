import json
import subprocess
import sys
from pathlib import Path

import pytest

from finance_ca_assistant.evaluation.golden_testset import GoldenQuestion
from finance_ca_assistant.evaluation.retrieval_benchmark import (
    check_gate,
    method_search_fns,
    render_markdown,
    run_benchmark,
)
from finance_ca_assistant.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


def chunk(source, page, references=()):
    return {"id": f"{source}-{page}", "source": source, "page": page, "references": list(references)}


QUESTIONS = [
    GoldenQuestion(id="q1", question="one", category="audit", correct_sources=["a.pdf"],
                   expected_sections=["audit_standard:sa_500"], expected_pages=["a.pdf#3"]),
    GoldenQuestion(id="q2", question="two", category="gst", correct_sources=["b.pdf"], expected_sections=[]),
]

RESULTS = {
    # q1: source at rank 2, gold page at rank 3, section in the rank-3 chunk.
    "one": [chunk("x.pdf", 1), chunk("a.pdf", 1), chunk("a.pdf", 3, ["audit_standard:sa_500"])],
    # q2: never found.
    "two": [chunk("x.pdf", 1)],
}


def fake_search(query, k):
    return RESULTS[query][:k]


def test_metrics_follow_ranks_and_only_count_labelled_questions():
    report = run_benchmark(QUESTIONS, {"hybrid": fake_search}, ks=(1, 3))
    metrics = report["methods"]["hybrid"]["metrics"]

    assert metrics["source_recall@1"] == 0.0
    assert metrics["source_recall@3"] == 0.5
    assert metrics["source_mrr@3"] == pytest.approx(0.5 / 2)
    # Page and section metrics are averaged over q1 only: q2 has no labels for them.
    assert metrics["page_recall@1"] == 0.0
    assert metrics["page_recall@3"] == 1.0
    assert metrics["page_mrr@3"] == pytest.approx(1 / 3)
    assert metrics["section_recall@1"] == 0.0
    assert metrics["section_recall@3"] == 1.0
    assert [m["id"] for m in report["misses"]] == ["q2"]
    assert set(report["hybrid_by_category"]) == {"audit", "gst"}


def test_questions_without_their_source_in_the_corpus_are_skipped_not_failed():
    report = run_benchmark(QUESTIONS, {"hybrid": fake_search}, corpus_sources=["a.pdf", "x.pdf"], ks=(1, 3))

    assert report["questions_scored"] == 1
    assert report["skipped"] == [{"id": "q2", "reason": "expected source not in corpus"}]
    assert report["methods"]["hybrid"]["metrics"]["source_recall@3"] == 1.0
    assert report["golden_ids"] == ["q1", "q2"]


def test_questions_whose_labelled_pages_were_not_chunked_are_skipped():
    # q1 is labelled against page 3, but a bounded build only kept page 1.
    report = run_benchmark(QUESTIONS, {"hybrid": fake_search}, corpus_sources=["a.pdf", "b.pdf"],
                           corpus_pages=["a.pdf#1", "b.pdf#1"], ks=(1, 3))

    assert report["skipped"] == [{"id": "q1", "reason": "expected pages not in corpus"}]
    assert report["questions_scored"] == 1
    # q2 has no page labels, so it is scored whatever pages exist.
    assert [m["id"] for m in report["misses"]] == ["q2"]


def test_gate_flags_drops_beyond_tolerance_and_changed_golden_sets():
    baseline = run_benchmark(QUESTIONS, {"hybrid": fake_search}, ks=(1, 3))
    assert check_gate(baseline, baseline) == []

    worse = json.loads(json.dumps(baseline))
    worse["methods"]["hybrid"]["metrics"]["source_recall@3"] = 0.49
    worse["methods"]["hybrid"]["metrics"]["page_recall@3"] = 0.9
    assert check_gate(worse, baseline, tolerance=0.02) == ["page_recall@3 fell from 1.000 to 0.900"]

    worse["golden_ids"] = ["q1"]
    assert "Golden set differs" in check_gate(worse, baseline)[0]


def test_benchmark_runs_every_method_on_a_real_pipeline(sample_chunks):
    pipeline = RAGPipeline.from_chunks(sample_chunks)
    questions = [GoldenQuestion(
        id="tax_audit_001", question="What does Form 3CD clause 44 require?", category="tax_audit",
        correct_sources=["ICAI_Tax_Audit_44AB_Guidance_Note_2023.pdf"], expected_sections=["form_3cd:44"],
        expected_pages=["ICAI_Tax_Audit_44AB_Guidance_Note_2023.pdf#221"],
    )]

    report = run_benchmark(questions, method_search_fns(pipeline))

    assert list(report["methods"]) == ["hybrid", "dense", "bm25", "clause"]
    hybrid = report["methods"]["hybrid"]["metrics"]
    assert hybrid["source_recall@1"] == 1.0
    assert hybrid["page_recall@1"] == 1.0
    assert hybrid["section_recall@1"] == 1.0
    markdown = render_markdown(report, gate_failures=[])
    assert "| hybrid |" in markdown and "- Passed" in markdown


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark_retrieval.py"), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_script_writes_a_baseline_then_gates_against_it(tmp_path, sample_chunks):
    artifacts = tmp_path / "artifacts"
    RAGPipeline.from_chunks(sample_chunks).save_artifacts(artifacts)
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{
        "id": "tax_audit_001", "question": "What does Form 3CD clause 44 require?", "category": "tax_audit",
        "correct_sources": ["ICAI_Tax_Audit_44AB_Guidance_Note_2023.pdf"], "expected_sections": ["form_3cd:44"],
    }]))
    baseline = tmp_path / "baseline.json"
    common = ["--artifacts", str(artifacts), "--golden", str(golden), "--baseline", str(baseline),
              "--out-dir", str(tmp_path / "out")]

    written = run_script(*common, "--write-baseline")
    assert written.returncode == 0, written.stderr
    assert run_script(*common).returncode == 0

    raised = json.loads(baseline.read_text())
    raised["methods"]["hybrid"]["metrics"]["section_recall@1"] = 2.0
    baseline.write_text(json.dumps(raised))
    gated = run_script(*common)
    assert gated.returncode == 1
    assert "FAIL: section_recall@1 fell from 2.000 to 1.000" in gated.stdout
    assert (tmp_path / "out" / "report.md").exists()
