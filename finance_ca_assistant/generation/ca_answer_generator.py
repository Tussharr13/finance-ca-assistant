"""Fail-closed extractive answers with source-page attribution."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from finance_ca_assistant.generation.ca_prompt_template import CA_SYSTEM_PROMPT, build_generation_prompt
from finance_ca_assistant.generation.citation_validator import CitationValidator
from finance_ca_assistant.generation.llm_factory import EchoLLM, LLMClient


class CAAnswerGenerator:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or EchoLLM()
        self.citation_validator = CitationValidator()

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_id, duplicates = {}, set()
        for chunk in retrieved_chunks:
            key = chunk.get("id")
            if not isinstance(key, str) or not key:
                continue
            if key in by_id:
                duplicates.add(key)
            by_id[key] = chunk
        errors, claims, accepted, sources = [], [], [], []
        parsed = None
        if not retrieved_chunks:
            errors.append("no_retrieved_context")
        elif isinstance(self.llm, EchoLLM):
            errors.append("generation_disabled")
        else:
            raw = self.llm.generate(CA_SYSTEM_PROMPT, build_generation_prompt(question, retrieved_chunks))
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                errors.append("invalid_json")
            if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list) or not 1 <= len(parsed["claims"]) <= 10:
                errors.append("expected_1_to_10_claims")
            else:
                claims = parsed["claims"]
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict) or any(
                not isinstance(claim.get(key), str) or not claim[key].strip()
                for key in ("text", "quote", "chunk_id")
            ):
                errors.append(f"claim_{index}:invalid_schema")
                continue
            key = claim["chunk_id"]
            if key not in by_id or key in duplicates:
                errors.append(f"claim_{index}:unknown_or_ambiguous_chunk")
                continue
            chunk = by_id[key]
            check = self.citation_validator.validate_extract(claim["text"], claim["quote"], chunk)
            if not check.valid:
                errors.append(f"claim_{index}:{check.reason}")
                continue
            accepted.append({"text": " ".join(claim["quote"].split()), "chunk_id": key})
            sources.append({"chunk_id": key, "document": chunk["source"], "page": chunk["page"], "quote": claim["quote"]})
        if errors:
            accepted, sources = [], []
        verified = bool(accepted)
        return {
            "status": "evidence_only" if verified else "abstained",
            "answer": "\n\n".join(f"[{i}] {claim['text']}" for i, claim in enumerate(accepted, 1)) if verified else "I cannot provide a citation-verified answer from the available output.",
            "claims": accepted,
            "basis": {
                "primary_sources": sources,
                "confidence": "not_assessed",
                "evidence_status": "exact_quotes_verified" if verified else "unverified",
                "confidence_reasons": ["Quote matching verifies attribution only.", "Relevance, completeness, current law, and client applicability are not validated."],
            },
            "ca_analysis": "Only source extracts are available; no financial conclusion has been validated.",
            "risks": ["Source extracts may be incomplete, outdated, or inapplicable."],
            "missing_info": "Current applicability and required client facts need verification.",
            "next_steps": ["Review full cited pages and confirm applicability."],
            "disclaimer": "Educational source evidence only.",
            "diagnostics": {"parsed_llm_json": isinstance(parsed, dict), "citation_errors": errors, "citation_policy": "exact_extract_v1"},
        }
