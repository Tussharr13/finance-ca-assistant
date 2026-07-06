"""CA RAG retrieval and generation metrics."""

from __future__ import annotations

from typing import Iterable, List


def mean_reciprocal_rank(retrieved_sources: List[List[str]], expected_sources: List[List[str]]) -> float:
    """Compute MRR for retrieved source IDs/names."""

    if not retrieved_sources:
        return 0.0
    total = 0.0
    for retrieved, expected in zip(retrieved_sources, expected_sources):
        expected_set = set(expected)
        rank_score = 0.0
        for rank, source in enumerate(retrieved, start=1):
            if source in expected_set:
                rank_score = 1.0 / rank
                break
        total += rank_score
    return total / len(retrieved_sources)


def source_coverage(retrieved_sources: List[List[str]], expected_sources: List[List[str]]) -> float:
    """Return share of questions where at least one expected source is retrieved."""

    if not retrieved_sources:
        return 0.0
    covered = 0
    for retrieved, expected in zip(retrieved_sources, expected_sources):
        if set(retrieved) & set(expected):
            covered += 1
    return covered / len(retrieved_sources)


def section_recall(retrieved_sections: List[List[str]], expected_sections: List[List[str]]) -> float:
    """Return average recall of expected sections."""

    recalls = []
    for retrieved, expected in zip(retrieved_sections, expected_sections):
        expected_set = set(expected)
        if not expected_set:
            recalls.append(1.0)
        else:
            recalls.append(len(set(retrieved) & expected_set) / len(expected_set))
    return sum(recalls) / len(recalls) if recalls else 0.0


def hallucination_rate(validity_flags: Iterable[bool]) -> float:
    """Return fraction of answers that contain hallucinated references."""

    flags = list(validity_flags)
    if not flags:
        return 0.0
    return sum(1 for flag in flags if not flag) / len(flags)

