"""Validation helpers for CA answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from finance_ca_assistant.domain_index import CADomainIndex, extract_references


@dataclass(frozen=True)
class ReferenceValidationReport:
    """Result of checking answer references against the domain index."""

    valid: bool
    valid_references: List[str]
    invalid_references: List[str]


def validate_answer_references(answer: str, index: CADomainIndex) -> ReferenceValidationReport:
    """Flag statutory/form references in an answer that do not exist in the KB."""
    valid_references: List[str] = []
    invalid_references: List[str] = []

    for reference in extract_references(answer):
        if index.reference_exists(reference.normalized):
            valid_references.append(reference.normalized)
        else:
            invalid_references.append(reference.normalized)

    return ReferenceValidationReport(
        valid=not invalid_references,
        valid_references=valid_references,
        invalid_references=invalid_references,
    )

