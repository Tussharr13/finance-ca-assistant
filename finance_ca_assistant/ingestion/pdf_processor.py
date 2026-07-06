"""PDF text, table, and structure extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from finance_ca_assistant.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class PageText:
    """Extracted text for one PDF page."""

    page: int
    text: str


@dataclass(frozen=True)
class ProcessedPDF:
    """Structured extraction result for one PDF."""

    source: str
    path: str
    pages: List[PageText] = field(default_factory=list)
    tables: List[Dict[str, object]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class PDFProcessor:
    """Extract text from PDFs and fail gracefully for encrypted/corrupt files.

    OCR and table extraction are optional extension points. In Kaggle's free
    environment, this processor stays lightweight and records when a scanned PDF
    likely needs OCR instead of failing silently.
    """

    def process(self, pdf_path: str | Path, max_pages: int | None = None) -> ProcessedPDF:
        """Extract readable text from a PDF.

        Args:
            pdf_path: PDF path to process.
            max_pages: Optional page cap for smoke tests and low-memory notebooks.
        """

        path = Path(pdf_path)
        try:
            from pypdf import PdfReader

            if max_pages is not None and max_pages <= 0:
                raise ValueError("max_pages must be positive when provided")

            reader = PdfReader(str(path))
            if reader.is_encrypted:
                return ProcessedPDF(
                    source=path.name,
                    path=str(path),
                    metadata={"encrypted": True, "is_scanned": False},
                    errors=["encrypted_pdf"],
                )

            pages: List[PageText] = []
            empty_pages = 0
            for page_number, page in enumerate(reader.pages, start=1):
                if max_pages is not None and page_number > max_pages:
                    break
                text = clean_extracted_text(page.extract_text() or "")
                if text:
                    pages.append(PageText(page=page_number, text=text))
                else:
                    empty_pages += 1

            processed_page_count = min(len(reader.pages), max_pages or len(reader.pages))
            is_scanned = bool(processed_page_count) and empty_pages == processed_page_count
            return ProcessedPDF(
                source=path.name,
                path=str(path),
                pages=pages,
                tables=[],
                metadata={
                    "encrypted": False,
                    "is_scanned": is_scanned,
                    "page_count": len(reader.pages),
                    "text_page_count": len(pages),
                    "processed_page_count": processed_page_count,
                    "max_pages": max_pages,
                },
                errors=["ocr_required"] if is_scanned else [],
            )
        except Exception as error:
            logger.warning("Failed to process PDF %s: %s", path, error)
            return ProcessedPDF(source=path.name, path=str(path), errors=[str(error)])


def clean_extracted_text(text: str) -> str:
    """Normalize extracted PDF text while preserving paragraph breaks."""

    lines = [line.strip() for line in text.splitlines()]
    compact: List[str] = []
    for line in lines:
        if line:
            compact.append(" ".join(line.split()))
        elif compact and compact[-1] != "":
            compact.append("")
    return "\n".join(compact).strip()
