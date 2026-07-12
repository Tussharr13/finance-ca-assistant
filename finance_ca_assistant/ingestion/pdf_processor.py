"""PDF text, table, and structure extraction."""

from __future__ import annotations

import gc
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

    PyMuPDF is preferred because it extracts pages without materializing the
    potentially enormous decoded content streams used by ``pypdf``. OCR and
    table extraction remain optional extension points.
    """

    def __init__(
        self,
        backend: str = "auto",
        max_page_text_chars: int = 500_000,
        cache_shrink_interval: int = 5,
    ) -> None:
        if backend not in {"auto", "pymupdf", "pypdf"}:
            raise ValueError("backend must be auto, pymupdf, or pypdf")
        if max_page_text_chars <= 0:
            raise ValueError("max_page_text_chars must be positive")
        if cache_shrink_interval <= 0:
            raise ValueError("cache_shrink_interval must be positive")
        self.backend = backend
        self.max_page_text_chars = max_page_text_chars
        self.cache_shrink_interval = cache_shrink_interval

    def process(self, pdf_path: str | Path, max_pages: int | None = None) -> ProcessedPDF:
        """Extract readable text from a PDF.

        Args:
            pdf_path: PDF path to process.
            max_pages: Optional page cap for smoke tests and low-memory notebooks.
        """

        if max_pages is not None and max_pages <= 0:
            raise ValueError("max_pages must be positive when provided")

        path = Path(pdf_path)
        try:
            if self.backend in {"auto", "pymupdf"}:
                try:
                    import pymupdf
                except ImportError:
                    if self.backend == "pymupdf":
                        raise RuntimeError(
                            "Install pymupdf to use the memory-bounded PDF backend"
                        )
                else:
                    return self._process_with_pymupdf(path, max_pages, pymupdf)

            return self._process_with_pypdf(path, max_pages)
        except Exception as error:
            logger.warning("Failed to process PDF %s: %s", path, error)
            return ProcessedPDF(source=path.name, path=str(path), errors=[str(error)])

    def _process_with_pymupdf(
        self,
        path: Path,
        max_pages: int | None,
        pymupdf: object,
    ) -> ProcessedPDF:
        pages: List[PageText] = []
        errors: List[str] = []
        empty_pages = 0
        document = pymupdf.open(str(path))
        try:
            if document.needs_pass:
                return _encrypted_result(path, backend="pymupdf")

            page_count = int(document.page_count)
            processed_page_count = min(page_count, max_pages or page_count)
            for page_index in range(processed_page_count):
                page = document.load_page(page_index)
                try:
                    text = clean_extracted_text(page.get_text("text", sort=True) or "")
                finally:
                    del page

                text, truncated = self._limit_page_text(text)
                if truncated:
                    errors.append(f"page_{page_index + 1}_text_truncated")
                if text:
                    pages.append(PageText(page=page_index + 1, text=text))
                else:
                    empty_pages += 1

                if (page_index + 1) % self.cache_shrink_interval == 0:
                    pymupdf.TOOLS.store_shrink(100)
                    gc.collect()
                    logger.info(
                        "Extracted %s/%s pages from %s; RSS=%s",
                        page_index + 1,
                        processed_page_count,
                        path.name,
                        format_process_rss(),
                    )

            is_scanned = bool(processed_page_count) and empty_pages == processed_page_count
            if is_scanned:
                errors.append("ocr_required")
            return _processed_result(
                path=path,
                pages=pages,
                errors=errors,
                page_count=page_count,
                processed_page_count=processed_page_count,
                max_pages=max_pages,
                is_scanned=is_scanned,
                backend="pymupdf",
            )
        finally:
            document.close()
            pymupdf.TOOLS.store_shrink(100)
            gc.collect()

    def _process_with_pypdf(self, path: Path, max_pages: int | None) -> ProcessedPDF:
        from pypdf import PdfReader

        logger.warning(
            "PyMuPDF is unavailable; using pypdf fallback for %s. "
            "Large decoded page streams may use substantial RAM.",
            path.name,
        )
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return _encrypted_result(path, backend="pypdf")

        pages: List[PageText] = []
        errors: List[str] = []
        empty_pages = 0
        page_count = len(reader.pages)
        processed_page_count = min(page_count, max_pages or page_count)
        for page_index in range(processed_page_count):
            text = clean_extracted_text(reader.pages[page_index].extract_text() or "")
            text, truncated = self._limit_page_text(text)
            if truncated:
                errors.append(f"page_{page_index + 1}_text_truncated")
            if text:
                pages.append(PageText(page=page_index + 1, text=text))
            else:
                empty_pages += 1

        is_scanned = bool(processed_page_count) and empty_pages == processed_page_count
        if is_scanned:
            errors.append("ocr_required")
        return _processed_result(
            path=path,
            pages=pages,
            errors=errors,
            page_count=page_count,
            processed_page_count=processed_page_count,
            max_pages=max_pages,
            is_scanned=is_scanned,
            backend="pypdf",
        )

    def _limit_page_text(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_page_text_chars:
            return text, False
        return text[: self.max_page_text_chars], True


def _encrypted_result(path: Path, backend: str) -> ProcessedPDF:
    return ProcessedPDF(
        source=path.name,
        path=str(path),
        metadata={"encrypted": True, "is_scanned": False, "backend": backend},
        errors=["encrypted_pdf"],
    )


def _processed_result(
    path: Path,
    pages: List[PageText],
    errors: List[str],
    page_count: int,
    processed_page_count: int,
    max_pages: int | None,
    is_scanned: bool,
    backend: str,
) -> ProcessedPDF:
    return ProcessedPDF(
        source=path.name,
        path=str(path),
        pages=pages,
        tables=[],
        metadata={
            "encrypted": False,
            "is_scanned": is_scanned,
            "page_count": page_count,
            "text_page_count": len(pages),
            "processed_page_count": processed_page_count,
            "max_pages": max_pages,
            "backend": backend,
        },
        errors=errors,
    )


def get_process_rss_mb() -> float | None:
    """Return current Linux resident memory in MiB when available."""

    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return None


def format_process_rss() -> str:
    rss_mb = get_process_rss_mb()
    return f"{rss_mb:.1f} MiB" if rss_mb is not None else "unavailable"


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
