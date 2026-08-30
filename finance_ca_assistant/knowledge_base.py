"""Knowledge-base build helpers for Kaggle and local runs."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from finance_ca_assistant.config import (
    AppConfig,
    DEFAULT_SOURCE_CATALOG,
    SourceDefinition,
    ensure_directories,
    load_config,
)
from finance_ca_assistant.ingestion.pdf_downloader import PDFDownloader
from finance_ca_assistant.ingestion.pdf_processor import PDFProcessor, ProcessedPDF, format_process_rss
from finance_ca_assistant.ingestion.source_registry import SourceRegistry
from finance_ca_assistant.logger import get_logger
from finance_ca_assistant.pipeline import RAGPipeline
from finance_ca_assistant.processing.metadata_builder import MetadataBuilder
from finance_ca_assistant.processing.semantic_chunker import SemanticChunker
from finance_ca_assistant.utils.pdf_utils import safe_pdf_name


logger = get_logger(__name__)
CACHE_SCHEMA_VERSION = 1
CHUNKER_SETTINGS = {"max_chars": 1200, "overlap": 150}


@dataclass(frozen=True)
class KnowledgeBaseBuildResult:
    """Result of a knowledge-base build."""

    pdf_paths: List[str]
    chunks: List[Dict[str, object]]
    failed_sources: List[Dict[str, str]] = field(default_factory=list)
    failed_pdfs: List[Dict[str, str]] = field(default_factory=list)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    reused_chunks: bool = False
    cache_manifest_path: str = ""


def chunks_cache_manifest_path(chunks_path: str | Path) -> Path:
    """Return the sidecar manifest path for a chunk cache."""

    return Path(chunks_path).with_name("chunks.manifest.json")


def _normalize_sources(
    sources: Mapping[str, str | SourceDefinition],
) -> List[SourceDefinition]:
    normalized: List[SourceDefinition] = []
    for source_id, source in sources.items():
        if isinstance(source, SourceDefinition):
            if source.source_id != source_id:
                raise ValueError(f"Source key {source_id!r} does not match {source.source_id!r}")
            normalized.append(source)
        else:
            normalized.append(
                SourceDefinition(
                    source_id=source_id,
                    title=source_id,
                    url=str(source),
                    authority="unknown",
                    document_type=infer_document_type(source_id),
                    version="unversioned",
                )
            )
    return normalized


def build_chunks_cache_spec(
    sources: Iterable[SourceDefinition],
    max_pages_per_pdf: Optional[int],
    max_chunks_per_source: Optional[int],
    pdf_backend: str,
) -> Dict[str, object]:
    """Build the canonical set of inputs that determine chunk-cache validity."""

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sources": [asdict(source) for source in sources],
        "max_pages_per_pdf": max_pages_per_pdf,
        "max_chunks_per_source": max_chunks_per_source,
        "pdf_backend": pdf_backend,
        "chunker": CHUNKER_SETTINGS,
    }


def _fingerprint(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_chunks_cache_manifest(
    chunks_path: str | Path,
    build_spec: Mapping[str, object],
    chunk_count: int,
) -> str:
    """Write a fingerprinted sidecar for a completed chunk cache."""

    cache_path = Path(chunks_path)
    manifest_path = chunks_cache_manifest_path(cache_path)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": _fingerprint(build_spec),
        "build_spec": dict(build_spec),
        "chunk_count": chunk_count,
        "chunks_sha256": _file_sha256(cache_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return str(manifest_path)


def _load_valid_cached_chunks(
    chunks_path: Path,
    build_spec: Mapping[str, object],
) -> Optional[List[Dict[str, object]]]:
    manifest_path = chunks_cache_manifest_path(chunks_path)
    if not chunks_path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != _fingerprint(build_spec):
            return None
        if manifest.get("chunks_sha256") != _file_sha256(chunks_path):
            return None
        chunks = load_chunks_jsonl(chunks_path)
        if manifest.get("chunk_count") != len(chunks):
            return None
        return chunks
    except (OSError, ValueError, json.JSONDecodeError):
        logger.warning("Invalid chunk-cache manifest at %s; rebuilding", manifest_path)
        return None


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
    max_chunks: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Convert extracted PDF pages into enriched chunk dictionaries."""

    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive when provided")

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
            if max_chunks is not None and len(chunks) >= max_chunks:
                return chunks

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


def find_prebuilt_chunks(
    search_roots: Iterable[str | Path],
    exclude: str | Path | None = None,
) -> Path | None:
    """Find a deterministic prebuilt ``chunks.jsonl`` under attached inputs."""

    excluded = Path(exclude).resolve() if exclude is not None else None
    candidates: List[Path] = []
    for search_root in search_roots:
        root = Path(search_root)
        if not root.exists():
            continue
        for candidate in root.rglob("chunks.jsonl"):
            if not candidate.is_file():
                continue
            if excluded is not None and candidate.resolve() == excluded:
                continue
            candidates.append(candidate)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def seed_chunks_cache(source_path: str | Path, target_path: str | Path) -> bool:
    """Validate and copy a prebuilt chunk artifact into the writable cache."""

    source = Path(source_path)
    target = Path(target_path)
    if target.exists() or source.resolve() == target.resolve():
        return False
    chunks = load_chunks_jsonl(source)
    if not chunks:
        raise ValueError(f"Prebuilt chunk artifact is empty: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    source_manifest = chunks_cache_manifest_path(source)
    if source_manifest.exists():
        shutil.copyfile(source_manifest, chunks_cache_manifest_path(target))
    logger.info("Seeded chunk cache with %s chunks from %s", len(chunks), source)
    return True


def build_chunks_from_pdf_paths(
    pdf_paths: Iterable[str | Path],
    max_pages_per_pdf: Optional[int] = None,
    max_chunks_per_source: Optional[int] = None,
    pdf_backend: str = "auto",
) -> tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    """Process local PDFs and return chunks plus failures."""

    if max_chunks_per_source is not None and max_chunks_per_source <= 0:
        raise ValueError("max_chunks_per_source must be positive when provided")

    processor = PDFProcessor(backend=pdf_backend)
    chunks: List[Dict[str, object]] = []
    failed_pdfs: List[Dict[str, str]] = []
    pdf_path_list = [Path(pdf_path) for pdf_path in pdf_paths]

    for position, pdf_path in enumerate(pdf_path_list, start=1):
        logger.info("Processing PDF %s/%s: %s", position, len(pdf_path_list), pdf_path.name)
        processed = processor.process(pdf_path, max_pages=max_pages_per_pdf)
        if processed.errors and not processed.pages:
            failed_pdfs.append({"path": str(pdf_path), "error": "; ".join(processed.errors)})
            continue
        pdf_chunks = chunks_from_processed_pdf(
            processed,
            max_chunks=max_chunks_per_source,
        )
        chunks.extend(pdf_chunks)
        logger.info(
            "Built %s chunks from %s (%s text pages); RSS=%s",
            len(pdf_chunks),
            pdf_path.name,
            processed.metadata.get("text_page_count", len(processed.pages)),
            format_process_rss(),
        )
        del processed
        del pdf_chunks
        gc.collect()

    return chunks, failed_pdfs


def build_knowledge_base_from_sources(
    config: Optional[AppConfig] = None,
    source_urls: Optional[Mapping[str, str | SourceDefinition]] = None,
    limit: Optional[int] = None,
    max_pages_per_pdf: Optional[int] = None,
    max_chunks_per_source: Optional[int] = None,
    pdf_backend: str = "auto",
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

    selected_sources = _normalize_sources(source_urls or DEFAULT_SOURCE_CATALOG)
    if limit is not None:
        selected_sources = selected_sources[:limit]
    build_spec = build_chunks_cache_spec(
        selected_sources,
        max_pages_per_pdf,
        max_chunks_per_source,
        pdf_backend,
    )

    if not rebuild:
        cached_chunks = _load_valid_cached_chunks(selected_config.chunks_path, build_spec)
        if cached_chunks is not None:
            logger.info(
                "Reusing %s cached chunks from %s",
                len(cached_chunks),
                selected_config.chunks_path,
            )
            return KnowledgeBaseBuildResult(
                pdf_paths=[],
                chunks=cached_chunks,
                reused_chunks=True,
                cache_manifest_path=str(chunks_cache_manifest_path(selected_config.chunks_path)),
            )
        logger.info("Chunk cache missing or stale; rebuilding %s", selected_config.chunks_path)

    downloader = PDFDownloader(selected_config.raw_dir)
    registry = SourceRegistry(selected_config.manifest_path)
    pdf_paths: List[str] = []
    failed_sources: List[Dict[str, str]] = []

    for source in selected_sources:
        try:
            result = downloader.download(
                source.url,
                target_name=safe_pdf_name(source.source_id),
                use_cache_on_failure=True,
                prefer_cache=not rebuild,
            )
            source_metadata = asdict(source)
            registry.register_source(
                source.source_id,
                result.path,
                version=source.version,
                expected_sha256=source.expected_sha256,
                metadata=source_metadata,
            )
            pdf_paths.append(str(result.path))
            logger.info("Ready source %s -> %s", source.source_id, result.path)
        except Exception as error:
            logger.warning("Failed source %s: %s", source.source_id, error)
            failed_sources.append(
                {"source_id": source.source_id, "url": source.url, "error": str(error)}
            )

    chunks, failed_pdfs = build_chunks_from_pdf_paths(
        pdf_paths,
        max_pages_per_pdf=max_pages_per_pdf,
        max_chunks_per_source=max_chunks_per_source,
        pdf_backend=pdf_backend,
    )
    write_chunks_jsonl(chunks, selected_config.chunks_path)
    cache_manifest_path = write_chunks_cache_manifest(
        selected_config.chunks_path,
        build_spec,
        len(chunks),
    )
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
        cache_manifest_path=cache_manifest_path,
    )
