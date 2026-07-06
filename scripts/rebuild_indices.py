"""Rebuild the clause index from a chunks JSONL file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finance_ca_assistant.processing.clause_extractor import ClauseExtractor


def main() -> None:
    chunks_path = Path("data/processed/chunks.jsonl")
    if not chunks_path.exists():
        raise SystemExit("No chunks found at data/processed/chunks.jsonl")
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line]
    clause_index = ClauseExtractor().build_index(chunks)
    output = Path("data/clause_index.json")
    output.write_text(json.dumps(clause_index, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

