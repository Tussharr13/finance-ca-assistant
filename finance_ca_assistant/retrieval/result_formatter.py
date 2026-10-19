"""Formatting helpers for retrieved chunks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def format_retrieval_results(results: Iterable[object], preview_chars: int = 240) -> str:
    """Format retrieval results for notebook or API diagnostics."""

    lines = []
    for rank, result in enumerate(results, start=1):
        chunk = getattr(result, "chunk", {})
        score = getattr(result, "score", 0.0)
        methods = ",".join(getattr(result, "methods", [getattr(result, "method", "unknown")]))
        preview = str(chunk.get("text") or "")[:preview_chars].replace("\n", " ")
        lines.append(
            f"{rank}. {chunk.get('source')} page {chunk.get('page')} | "
            f"score={score:.3f} | methods={methods}\n{preview}..."
        )
    return "\n\n".join(lines)



def source_payloads(results: Iterable[object], snippet_chars: int = 320) -> List[Dict[str, Any]]:
    """Structured retrieval results for API clients: one entry per chunk, in rank order."""

    payloads = []
    for rank, result in enumerate(results, start=1):
        chunk = getattr(result, "chunk", {})
        text = " ".join(str(chunk.get("text") or "").split())
        payloads.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("id"),
                "document": chunk.get("source"),
                "page": chunk.get("page"),
                "snippet": text[:snippet_chars] + ("…" if len(text) > snippet_chars else ""),
                "methods": list(getattr(result, "methods", [getattr(result, "method", "unknown")])),
            }
        )
    return payloads
