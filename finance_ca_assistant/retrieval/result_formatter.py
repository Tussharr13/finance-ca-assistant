"""Formatting helpers for retrieved chunks."""

from __future__ import annotations

from typing import Iterable


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

