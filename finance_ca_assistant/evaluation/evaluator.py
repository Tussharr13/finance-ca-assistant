"""Evaluation runner for retrieval quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from finance_ca_assistant.domain_index import extract_references
from finance_ca_assistant.evaluation.ca_metrics import mean_reciprocal_rank, section_recall, source_coverage
from finance_ca_assistant.evaluation.golden_testset import GoldenQuestion


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation result payload."""

    metrics: Dict[str, float]
    failures: List[Dict[str, object]] = field(default_factory=list)


class EvaluationRunner:
    """Run golden-test retrieval evaluations."""

    def __init__(self, questions: List[GoldenQuestion]) -> None:
        self.questions = questions

    def evaluate_retrieval(self, retrieve_fn: Callable[[str], List[Dict[str, object]]]) -> EvaluationReport:
        retrieved_sources: List[List[str]] = []
        expected_sources: List[List[str]] = []
        retrieved_sections: List[List[str]] = []
        expected_sections: List[List[str]] = []
        failures: List[Dict[str, object]] = []

        for question in self.questions:
            chunks = retrieve_fn(question.question)
            sources = [str(chunk.get("source") or "") for chunk in chunks]
            sections = []
            for chunk in chunks:
                sections.extend(
                    reference.normalized for reference in extract_references(str(chunk.get("text") or ""))
                )
            retrieved_sources.append(sources)
            expected_sources.append(question.correct_sources)
            retrieved_sections.append(sections)
            expected_sections.append(question.expected_sections)
            if not (set(sources) & set(question.correct_sources)):
                failures.append({"id": question.id, "reason": "missing_expected_source"})

        return EvaluationReport(
            metrics={
                "mrr": mean_reciprocal_rank(retrieved_sources, expected_sources),
                "source_coverage": source_coverage(retrieved_sources, expected_sources),
                "section_recall": section_recall(retrieved_sections, expected_sections),
            },
            failures=failures,
        )

