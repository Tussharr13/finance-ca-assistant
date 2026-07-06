"""Detect invalid CA-domain references in generated answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from finance_ca_assistant.domain_index import CADomainIndex
from finance_ca_assistant.validators import validate_answer_references


@dataclass(frozen=True)
class HallucinationReport:
    """Reference hallucination report."""

    valid: bool
    valid_references: List[str] = field(default_factory=list)
    invalid_references: List[str] = field(default_factory=list)


class HallucinationDetector:
    """Detect statutory/form references not present in the current KB."""

    def __init__(self, domain_index: CADomainIndex) -> None:
        self.domain_index = domain_index

    @classmethod
    def from_chunks(cls, chunks: List[Dict[str, Any]]) -> "HallucinationDetector":
        return cls(CADomainIndex.from_chunks(chunks))

    def detect(self, text: str) -> HallucinationReport:
        report = validate_answer_references(text, self.domain_index)
        return HallucinationReport(
            valid=report.valid,
            valid_references=report.valid_references,
            invalid_references=report.invalid_references,
        )

