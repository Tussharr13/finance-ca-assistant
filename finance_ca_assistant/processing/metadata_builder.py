"""Metadata enrichment for CA-domain chunks."""

from __future__ import annotations

from typing import Any, Dict

from finance_ca_assistant.domain_index import extract_references


class MetadataBuilder:
    """Attach source-family and reference metadata to chunks."""

    def enrich_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(chunk)
        source = str(enriched.get("source") or "")
        references = [reference.normalized for reference in extract_references(str(enriched.get("text") or ""))]
        enriched["references"] = references
        enriched["source_family"] = infer_source_family(source)
        return enriched


def infer_source_family(source: str) -> str:
    upper = source.upper()
    if "ICAI" in upper:
        return "ICAI"
    if "CBIC" in upper or "CGST" in upper:
        return "CBIC"
    if "INCOME" in upper:
        return "IncomeTax"
    return "Unknown"

