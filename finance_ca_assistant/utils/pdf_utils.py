"""PDF utility helpers."""

from __future__ import annotations

from pathlib import Path


def is_probably_pdf(path: str | Path) -> bool:
    """Return whether a file starts with the PDF header."""

    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size < 4:
        return False
    return file_path.read_bytes()[:4] == b"%PDF"


def safe_pdf_name(source_id: str) -> str:
    """Convert a source ID to a stable PDF filename."""

    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in source_id)
    return f"{cleaned}.pdf"

