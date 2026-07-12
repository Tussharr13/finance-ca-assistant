"""Knowledge-base build helpers for Kaggle and local runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from finance_ca_assistant.config import AppConfig, DEFAULT_SOURCE_URLS, ensure_directories, load_config
from finance_ca_assistant.ingestion.pdf_downloader import PDFDownloader
from finance_ca_assistant.ingestion.pdf_processor import PDFProcessor, ProcessedPDF
from finance_ca_assistant.ingestion.source_registry import SourceRegistry
from finance_ca_assistant.logger import get_logger
from finance_ca_assistant.pipeline import RAGPipeline
from finance_ca_assistant.processing.metadata_builder import MetadataBuilder
from finance_ca_assistant.processing.semantic_chunker import SemanticChunker
from finance_ca_assistant.utils.pdf_utils import safe_pdf_name


logger = get_logger(__name__)


@dataclass(frozen=True)
class KnowledgeBaseBuildResult:
    """Result of a knowledge-base build."""

    pdf_paths: List[str]
    chunks: List[Dict[str, object]]
    failed_sources: List[Dict[str, str]] = field(default_factory=list)
    failed_pdfs: List[Dict[str, str]] = field(default_factory=list)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    reused_chunks: bool = False


def infer_document_type(source_name: str) -> str:
    """Infer CA document type from a source filename or ID."""

    normalized = source_name.lower()
    if "tax_audit" in normalized or "44ab" in normalized or "3cd" in normalized:
        return "form_3cd"
    if "cgst" in normalized or "gst" in normalized or "rules" in normalized:
        return "gst_rules"
    if "_as_" in normalized or "accounting" in normalized or "ind_as" in normalized:
        return "accounting_standard"
    if "_sa_" in normalized or "audit_evidence" in normalized or "overall_objectives" in normalized:
        return "audit_standard"
    return "generic"


def chunks_from_processed_pdf(
    processed_pdf: ProcessedPDF,
    chunker: Optional[SemanticChunker] = None,
    metadata_builder: Optional[MetadataBuilder] = None,
) -> List[Dict[str, object]]:
    """Convert extracted PDF pages into enriched chunk dictionaries."""

    selected_chunker = chunker or SemanticChunker(max_chars=1200, overlap=150)
    selected_metadata_builder = metadata_builder or MetadataBuilder()
    document_type = infer_document_type(processed_pdf.source)
    chunks: List[Dict[str, object]] = []

    for page in processed_pdf.pages:
        text_chunks = selected_chunker.chunk(page.text, document_type=document_type)
        for chunk_index, chunk in enumerate(text_chunks, start=1):
            record: Dict[str, object] = {
                "id": f"{processed_pdf.source}::p{page.page:03d}::c{chunk_index:03d}",
                "source": processed_pdf.source,
                "page": page.page,
                "text": chunk.text,
                "document_type": document_type,
            }
            chunks.append(selected_metadata_builder.enrich_chunk(record))

    return chunks


def write_chunks_jsonl(chunks: Iterable[Mapping[str, object]], output_path: str | Path) -> str:
    """Write chunks to JSONL and return the output path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(dict(chunk), sort_keys=True) + "\n")
    return str(path)


def load_chunks_jsonl(path: str | Path) -> List[Dict[str, object]]:
    """Load chunk dictionaries from a JSONL artifact."""

    chunk_path = Path(path)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk cache not found: {chunk_path}")

    chunks: List[Dict[str, object]] = []
    with chunk_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid chunk JSON at {chunk_path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Chunk cache row must be an object at {chunk_path}:{line_number}"
                )
            chunks.append(chunk)
    return chunks


def build_chunks_from_pdf_paths(
    pdf_paths: Iterable[str | Path],
    max_pages_per_pdf: Optional[int] = None,
    max_chunks_per_source: Optional[int] = None,
) -> tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    """Process local PDFs and return chunks plus failures."""

    if max_chunks_per_source is not None and max_chunks_per_source <= 0:
        raise ValueError("max_chunks_per_source must be positive when provided")

    processor = PDFProcessor()
    chunks: List[Dict[str, object]] = []
    failed_pdfs: List[Dict[str, str]] = []
    pdf_path_list = [Path(pdf_path) for pdf_path in pdf_paths]

    for position, pdf_path in enumerate(pdf_path_list, start=1):
        logger.info("Processing PDF %s/%s: %s", position, len(pdf_path_list), pdf_path.name)
        processed = processor.process(pdf_path, max_pages=max_pages_per_pdf)
        if processed.errors and not processed.pages:
            failed_pdfs.append({"path": str(pdf_path), "error": "; ".join(processed.errors)})
            continue
        pdf_chunks = chunks_from_processed_pdf(processed)
        if max_chunks_per_source is not None:
            pdf_chunks = pdf_chunks[:max_chunks_per_source]
        chunks.extend(pdf_chunks)
        logger.info(
            "Built %s chunks from %s (%s text pages)",
            len(pdf_chunks),
            pdf_path.name,
            processed.metadata.get("text_page_count", len(processed.pages)),
        )

    return chunks, failed_pdfs


def build_knowledge_base_from_sources(
    config: Optional[AppConfig] = None,
    source_urls: Optional[Mapping[str, str]] = None,
    limit: Optional[int] = None,
    max_pages_per_pdf: Optional[int] = None,
    max_chunks_per_source: Optional[int] = None,
    rebuild: bool = True,
    build_artifacts: bool = True,
) -> KnowledgeBaseBuildResult:
    """Download authoritative PDFs, process them, and save RAG artifacts.

    This is the main Kaggle entry point. When ``rebuild`` is false, an existing
    ``chunks.jsonl`` is returned before any source download or PDF processing.
    Artifact construction can be disabled so ingestion and model indexing happen
    in separate notebook stages.
    """

    selected_config = config or load_config()
    ensure_directories(selected_config)

    if not rebuild and selected_config.chunks_path.exists():
        cached_chunks = load_chunks_jsonl(selected_config.chunks_path)
        if cached_chunks:
            logger.info(
                "Reusing %s cached chunks from %s",
                len(cached_chunks),
                selected_config.chunks_path,
            )
            return KnowledgeBaseBuildResult(
                pdf_paths=[],
                chunks=cached_chunks,
                reused_chunks=True,
            )
        logger.warning("Chunk cache is empty; rebuilding %s", selected_config.chunks_path)

    selected_sources = list((source_urls or DEFAULT_SOURCE_URLS).items())
    if limit is not None:
        selected_sources = selected_sources[:limit]

    downloader = PDFDownloader(selected_config.raw_dir)
    registry = SourceRegistry(selected_config.manifest_path)
    pdf_paths: List[str] = []
    failed_sources: List[Dict[str, str]] = []

    for source_id, url in selected_sources:
        try:
            result = downloader.download(url, target_name=safe_pdf_name(source_id), use_cache_on_failure=True)
            registry.register_source(source_id, result.path)
            pdf_paths.append(str(result.path))
            logger.info("Ready source %s -> %s", source_id, result.path)
        except Exception as error:
            logger.warning("Failed source %s: %s", source_id, error)
            failed_sources.append({"source_id": source_id, "url": url, "error": str(error)})

    chunks, failed_pdfs = build_chunks_from_pdf_paths(
        pdf_paths,
        max_pages_per_pdf=max_pages_per_pdf,
        max_chunks_per_source=max_chunks_per_source,
    )
    write_chunks_jsonl(chunks, selected_config.chunks_path)
    artifact_paths: Dict[str, str] = {}
    if chunks and build_artifacts:
        pipeline = RAGPipeline.from_chunks(chunks)
        artifact_paths = pipeline.save_artifacts(selected_config.indices_dir)

    return KnowledgeBaseBuildResult(
        pdf_paths=pdf_paths,
        chunks=chunks,
        failed_sources=failed_sources,
        failed_pdfs=failed_pdfs,
        artifact_paths=artifact_paths,
    )
