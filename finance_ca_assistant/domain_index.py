"""CA-domain reference extraction and direct clause/section indexing."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List


@dataclass(frozen=True)
class Reference:
    """A statutory, form, rule, standard, or audit reference found in text."""

    kind: str
    value: str
    normalized: str


@dataclass(frozen=True)
class ReferenceHit:
    """A chunk that mentions a normalized CA-domain reference."""

    chunk_id: str
    source: str
    page: int
    reference: str


REFERENCE_PATTERNS = (
    (
        "form_3cd",
        re.compile(
            r"\b(?:Form\s*)?3CD(?:\s+clause)?\s*(?:clause\s*)?[\(:]?\s*([0-9]{1,3}[A-Za-z]?)\)?",
            re.IGNORECASE,
        ),
    ),
    (
        "form_3cd",
        re.compile(r"\bClause\s+([0-9]{1,3}[A-Za-z]?)\s+of\s+Form\s+3CD\b", re.IGNORECASE),
    ),
    (
        "income_tax_section",
        re.compile(r"\b(?:Section|Sec\.?|u/s\.?)\s+([0-9]{1,3}[A-Za-z]?(?:\([a-zA-Z0-9]+\))*)", re.IGNORECASE),
    ),
    (
        "gst_rule",
        re.compile(r"\bRule\s+([0-9]{1,3}[A-Za-z]?)\b(?:\s+of\s+(?:the\s+)?CGST\s+Rules)?", re.IGNORECASE),
    ),
    (
        "accounting_standard",
        re.compile(r"\b(?:Ind\s+)?AS\s+([0-9]{1,3})\b", re.IGNORECASE),
    ),
    (
        "audit_standard",
        re.compile(r"\bSA\s+([0-9]{3})\b", re.IGNORECASE),
    ),
)


def extract_references(text: str) -> List[Reference]:
    """Extract CA-domain references from text in normalized form."""
    references: List[Reference] = []
    seen = set()

    for kind, pattern in REFERENCE_PATTERNS:
        for match in pattern.finditer(text or ""):
            raw_value = match.group(1).strip()
            normalized = normalize_reference(kind, raw_value)
            if normalized not in seen:
                references.append(Reference(kind=kind, value=raw_value, normalized=normalized))
                seen.add(normalized)

    return references


def normalize_reference(kind: str, value: str) -> str:
    clean_value = re.sub(r"\s+", "", value).lower()
    if kind == "accounting_standard":
        clean_value = f"as_{clean_value}"
    elif kind == "audit_standard":
        clean_value = f"sa_{clean_value}"
    return f"{kind}:{clean_value}"


class CADomainIndex:
    """Direct lookup index for Form 3CD clauses, sections, rules, AS, and SA refs."""

    def __init__(self) -> None:
        self._hits_by_reference: DefaultDict[str, List[ReferenceHit]] = defaultdict(list)

    @classmethod
    def from_chunks(cls, chunks: Iterable[Dict[str, Any]]) -> "CADomainIndex":
        index = cls()
        for fallback_index, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("id") or f"chunk-{fallback_index}")
            source = str(chunk.get("source") or "")
            page = int(chunk.get("page") or 0)
            text = str(chunk.get("text") or "")

            for reference in extract_references(text):
                hit = ReferenceHit(
                    chunk_id=chunk_id,
                    source=source,
                    page=page,
                    reference=reference.normalized,
                )
                if hit not in index._hits_by_reference[reference.normalized]:
                    index._hits_by_reference[reference.normalized].append(hit)

        return index

    def search_by_reference(self, reference_text: str) -> List[ReferenceHit]:
        references = extract_references(reference_text)
        if not references and ":" in reference_text:
            return list(self._hits_by_reference.get(reference_text.lower(), []))

        hits: List[ReferenceHit] = []
        for reference in references:
            hits.extend(self._hits_by_reference.get(reference.normalized, []))
        return hits

    def reference_exists(self, reference_text: str) -> bool:
        return bool(self.search_by_reference(reference_text))

    def references(self) -> List[str]:
        return sorted(self._hits_by_reference)

