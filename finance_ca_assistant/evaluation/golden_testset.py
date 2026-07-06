"""Golden CA-domain test set definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class GoldenQuestion:
    """One curated evaluation question."""

    id: str
    question: str
    category: str
    correct_sources: List[str]
    expected_sections: List[str]
    difficulty: str = "medium"
    correct_answer: str = ""


DEFAULT_GOLDEN_QUESTIONS = [
    GoldenQuestion(
        id="tax_audit_001",
        question="What does Form 3CD clause 44 require?",
        category="tax_audit",
        correct_sources=["ICAI_Tax_Audit_44AB_Guidance_Note_2023.pdf"],
        expected_sections=["form_3cd:44"],
        difficulty="easy",
    ),
    GoldenQuestion(
        id="audit_001",
        question="What does SA 500 say about audit evidence?",
        category="audit",
        correct_sources=["ICAI_SA_500_Audit_Evidence.pdf"],
        expected_sections=["audit_standard:sa_500"],
        difficulty="easy",
    ),
]


def load_golden_testset(path: str | Path) -> List[GoldenQuestion]:
    """Load golden questions from JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return [GoldenQuestion(**row) for row in rows]

