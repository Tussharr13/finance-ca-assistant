"""Generate synthetic CA search questions for chunks, with resumable checkpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

QUERY_SYSTEM_PROMPT = (
    "You write search questions that an Indian chartered accountant, auditor, or "
    "tax professional would type into a research tool. Questions must be specific, "
    "self-contained, and answerable from the given passage alone."
)

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_GENERIC = re.compile(r"\b(?:this|the|above|given)\s+(?:passage|text|document|section|excerpt)\b", re.IGNORECASE)


def build_query_prompt(chunk: Mapping[str, Any], questions_per_chunk: int, max_chars: int = 1500) -> str:
    text = " ".join(str(chunk.get("text") or "").split())[:max_chars]
    return (
        f"Source: {chunk.get('source')} (page {chunk.get('page')})\n"
        f"Passage:\n{text}\n\n"
        f"Write {questions_per_chunk} different questions this passage answers. "
        "Use the terms a professional would search for (section, clause, standard, form). "
        "Do not copy sentences from the passage and do not refer to 'the passage'. "
        'Return only a JSON array of strings. /no_think'
    )


def is_eligible_chunk(chunk: Mapping[str, Any], min_chars: int = 200) -> bool:
    """Skip chunks too short or too noisy to produce meaningful questions."""

    text = " ".join(str(chunk.get("text") or "").split())
    if len(text) < min_chars or not chunk.get("id"):
        return False
    letters = sum(character.isalpha() for character in text)
    if letters / len(text) < 0.6:
        return False
    # Tables of contents and indexes: dot leaders and page-number runs.
    if text.count("....") >= 3:
        return False
    return True


def parse_questions(raw: str) -> List[str]:
    """Read a JSON array of questions, falling back to one question per line."""

    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if isinstance(item, str) and item.strip()]
    lines = [_LIST_MARKER.sub("", line).strip().strip('"') for line in raw.splitlines()]
    return [line for line in lines if line.endswith("?")]


def filter_questions(questions: Iterable[str], chunk_text: str, seen: set) -> List[str]:
    """Drop short, generic, copied, or duplicate questions. Updates ``seen``."""

    normalized_chunk = " ".join(chunk_text.lower().split())
    kept: List[str] = []
    for question in questions:
        clean = " ".join(question.split())
        key = clean.lower().rstrip("?")
        if not 15 <= len(clean) <= 250 or len(clean.split()) < 4:
            continue
        if _GENERIC.search(clean) or key in seen:
            continue
        if key in normalized_chunk:
            continue
        seen.add(key)
        kept.append(clean)
    return kept


def generate_queries(
    chunks: Sequence[Mapping[str, Any]],
    llm: Any,
    output_path: str | Path,
    questions_per_chunk: int = 3,
    batch_size: int = 8,
    max_tokens: int = 200,
    max_chunks: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Generate questions for eligible chunks and append them to a JSONL checkpoint.

    Each line is ``{"chunk_id", "source", "page", "questions"}``. Chunks already in
    the file are skipped, so a disconnected Kaggle session resumes where it stopped.
    Uses ``llm.generate_batch`` when available, otherwise one call per chunk.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = load_query_records(path) if path.exists() else []
    done = {record["chunk_id"] for record in records}
    seen = {question.lower().rstrip("?") for record in records for question in record["questions"]}

    pending = [chunk for chunk in chunks if is_eligible_chunk(chunk) and chunk["id"] not in done]
    if max_chunks is not None:
        pending = pending[: max(0, max_chunks - len(done))]

    with path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            prompts = [build_query_prompt(chunk, questions_per_chunk) for chunk in batch]
            if hasattr(llm, "generate_batch"):
                outputs = llm.generate_batch(QUERY_SYSTEM_PROMPT, prompts, max_tokens=max_tokens, batch_size=batch_size)
            else:
                outputs = [llm.generate(QUERY_SYSTEM_PROMPT, prompt, max_tokens=max_tokens) for prompt in prompts]
            for chunk, raw in zip(batch, outputs):
                questions = filter_questions(parse_questions(raw), str(chunk.get("text") or ""), seen)
                record = {
                    "chunk_id": chunk["id"],
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "questions": questions[:questions_per_chunk],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
            handle.flush()
            if progress:
                progress(min(start + batch_size, len(pending)), len(pending))
    return records


def load_query_records(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
