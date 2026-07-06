"""Prompt templates for grounded CA-domain answers."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable


CA_SYSTEM_PROMPT = """You are an expert Indian Chartered Accountant (CA) assistant.

Critical rules:
1. Use only retrieved documents for document-specific claims.
2. Never invent section numbers, rule numbers, form fields, dates, or figures.
3. If the retrieved context is insufficient, say so clearly.
4. Every statutory or standard-specific claim must be traceable to a source.
5. Return structured JSON with answer, basis, ca_analysis, risks, missing_info,
   next_steps, and disclaimer.
"""


def build_generation_prompt(question: str, chunks: Iterable[Dict[str, Any]]) -> str:
    """Build a user prompt containing retrieved source context."""

    context_blocks = []
    for chunk in chunks:
        context_blocks.append(
            {
                "document": chunk.get("source"),
                "page": chunk.get("page"),
                "text": str(chunk.get("text") or "")[:1800],
                "references": chunk.get("references", []),
            }
        )
    payload = {
        "question": question,
        "retrieved_context": context_blocks,
        "required_output": {
            "answer": "Direct answer grounded in retrieved context",
            "basis": {"primary_sources": [], "confidence": "high|medium|low"},
            "ca_analysis": "Professional CA-style implications",
            "risks": [],
            "missing_info": "Missing information or not applicable",
            "next_steps": [],
            "disclaimer": "Educational guidance only; consult a qualified CA for filing decisions.",
        },
    }
    return json.dumps(payload, indent=2)

