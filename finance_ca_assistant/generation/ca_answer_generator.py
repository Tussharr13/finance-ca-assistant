"""Structured CA answer generation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from finance_ca_assistant.generation.ca_prompt_template import CA_SYSTEM_PROMPT, build_generation_prompt
from finance_ca_assistant.generation.hallucination_detector import HallucinationDetector
from finance_ca_assistant.generation.llm_factory import EchoLLM, LLMClient


class CAAnswerGenerator:
    """Generate structured answers with traceability fields."""

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or EchoLLM()

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a conservative structured CA answer.

        The local default remains conservative. With a real LLM configured,
        this method parses the generated JSON/text and returns it as the answer
        while preserving retrieved citations and hallucination diagnostics.
        """

        prompt = build_generation_prompt(question, retrieved_chunks)
        raw_response = self.llm.generate(CA_SYSTEM_PROMPT, prompt)
        primary_sources = [_source_payload(chunk) for chunk in retrieved_chunks[:5]]
        confidence = _confidence_from_sources(retrieved_chunks)
        detector = HallucinationDetector.from_chunks(retrieved_chunks)
        hallucination_report = detector.detect(raw_response)
        parsed_response = _parse_llm_response(raw_response)

        if parsed_response:
            answer_text = str(parsed_response.get("answer") or "").strip()
            ca_analysis = str(parsed_response.get("ca_analysis") or "").strip()
            risks = _list_field(parsed_response.get("risks"))
            missing_info = str(parsed_response.get("missing_info") or "").strip()
            next_steps = _list_field(parsed_response.get("next_steps"))
            disclaimer = str(parsed_response.get("disclaimer") or "").strip()
        elif not isinstance(self.llm, EchoLLM):
            answer_text = _clean_model_text(raw_response) or "The model did not return a usable answer."
            ca_analysis = "Use the retrieved source text as the primary basis and verify facts before filing or reporting."
            risks = ["Generated guidance must be checked against the cited source pages and current law."]
            missing_info = "Review whether the retrieved context contains all facts required for the conclusion."
            next_steps = ["Review cited pages", "Confirm applicability with current law and client facts"]
            disclaimer = "Educational guidance only; consult a qualified CA for filing, audit, or litigation decisions."
        else:
            answer_text = (
                "Based on the retrieved documents, the answer should be read with the cited "
                "source pages below. No document-specific claim should be used beyond this context."
            )
            ca_analysis = "Use the cited source text as the primary basis and verify facts before filing or reporting."
            risks = ["Incorrect source selection can lead to wrong statutory interpretation."]
            missing_info = "Not applicable based on the retrieved context." if retrieved_chunks else "No retrieved context was available."
            next_steps = ["Review cited pages", "Confirm applicability with current law and client facts"]
            disclaimer = "This is educational guidance. Consult a qualified CA for filing, audit, or litigation decisions."

        if not retrieved_chunks:
            answer_text = "Based on available documents, I cannot confirm the answer."
            missing_info = "No retrieved context was available."

        if not hallucination_report.valid:
            risks = list(risks) + [
                "The model output referenced unsupported sections or clauses; rely only on validated citations below."
            ]

        return {
            "answer": answer_text,
            "basis": {"primary_sources": primary_sources, "confidence": confidence},
            "ca_analysis": ca_analysis or "Use the cited source text as the primary basis and verify facts before filing or reporting.",
            "risks": risks,
            "missing_info": missing_info or "Not applicable based on the retrieved context.",
            "next_steps": next_steps,
            "disclaimer": disclaimer or "Educational guidance only; consult a qualified CA for filing, audit, or litigation decisions.",
            "diagnostics": {
                "raw_response_preview": raw_response[:500],
                "hallucination_valid": hallucination_report.valid,
                "invalid_references": hallucination_report.invalid_references,
                "parsed_llm_json": bool(parsed_response),
            },
        }


def _source_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    text = str(chunk.get("text") or "")
    return {
        "document": chunk.get("source"),
        "page": chunk.get("page"),
        "section": (chunk.get("references") or [""])[0] if chunk.get("references") else "",
        "quote": text[:200],
    }


def _confidence_from_sources(chunks: List[Dict[str, Any]]) -> str:
    if len(chunks) >= 3:
        return "high"
    if chunks:
        return "medium"
    return "low"


def _parse_llm_response(raw_response: str) -> Optional[Dict[str, Any]]:
    cleaned = _clean_model_text(raw_response)
    candidates = [cleaned]
    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        candidates.insert(0, json_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _clean_model_text(raw_response: str) -> str:
    text = str(raw_response or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _list_field(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []
