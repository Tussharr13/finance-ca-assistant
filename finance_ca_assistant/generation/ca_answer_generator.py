"""Fail-closed answers: paraphrased claims, each backed by a verified quote.

See decisionsForCA.md D-25. One invalid claim makes the whole answer abstain.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from finance_ca_assistant.generation.ca_prompt_template import CA_SYSTEM_PROMPT, build_generation_prompt
from finance_ca_assistant.generation.citation_validator import CitationValidator
from finance_ca_assistant.generation.llm_factory import EchoLLM, LLMClient


MAX_CLAIMS = 6


def _render(claims: List[Dict[str, str]], sources: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{i}] {claim['text']}\n    Source: {source['document']}, page {source['page']}: \"{source['quote']}\""
        for i, (claim, source) in enumerate(zip(claims, sources), 1)
    )


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
            # Models often wrap JSON in prose or code fences; take the outermost object.
            match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
            try:
                parsed = json.loads(match.group(0)) if match else None
            except (ValueError, TypeError):
                parsed = None
            if parsed is None:
                errors.append("invalid_json")
            if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list) or not 1 <= len(parsed["claims"]) <= MAX_CLAIMS:
                errors.append(f"expected_1_to_{MAX_CLAIMS}_claims")
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
            check = self.citation_validator.validate_quoted_claim(claim["text"], claim["quote"], chunk)
            if not check.valid:
                errors.append(f"claim_{index}:{check.reason}")
                continue
            quote = " ".join(claim["quote"].split())
            accepted.append({"text": " ".join(claim["text"].split()), "chunk_id": key, "quote": quote})
            sources.append({"chunk_id": key, "document": chunk["source"], "page": chunk["page"], "quote": quote})
        if errors:
            accepted, sources = [], []
        verified = bool(accepted)
        return {
            "status": "answered" if verified else "abstained",
            "answer": _render(accepted, sources) if verified else "I cannot provide a citation-verified answer from the available output.",
            "claims": accepted,
            "basis": {
                "primary_sources": sources,
                "confidence": "not_assessed",
                "evidence_status": "quotes_verified" if verified else "unverified",
                "confidence_reasons": [
                    "Each statement is checked against a verbatim quote: sentence boundaries, negations, conditions, numbers and wording overlap.",
                    "Relevance, completeness, current law, and client applicability are not validated.",
                ],
            },
            "ca_analysis": "Statements summarise the quoted sources; no financial conclusion has been validated.",
            "risks": ["Source extracts may be incomplete, outdated, or inapplicable."],
            "missing_info": "Current applicability and required client facts need verification.",
            "next_steps": ["Review full cited pages and confirm applicability."],
            "disclaimer": "Educational source evidence only.",
            "diagnostics": {"parsed_llm_json": isinstance(parsed, dict), "citation_errors": errors, "citation_policy": "quoted_claims_v1"},
        }
