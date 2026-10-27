"""Build the bounded benchmark corpus locally, matching the notebook's defaults.

    python3 scripts/build_corpus.py /path/to/artifacts
    python3 scripts/benchmark_retrieval.py --artifacts /path/to/artifacts

Downloads the first SOURCE_LIMIT catalog PDFs, keeps MAX_PAGES_PER_PDF pages of
each, chunks them, embeds with the notebook's embedding model and saves
chunks.jsonl, embeddings_cache.npz and clause_index.json. PDFs and the chunk
cache go under <artifacts>/build and are reused on rerun.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finance_ca_assistant.config import DEFAULT_SOURCE_CATALOG, AppConfig
from finance_ca_assistant.embeddings.embedding_factory import create_embedding_provider
from finance_ca_assistant.knowledge_base import build_knowledge_base_from_sources
from finance_ca_assistant.pipeline import RAGPipeline

# Same bounds as notebooks/finance_ca_assistant_kaggle_mvp.ipynb Section 3.
SOURCE_LIMIT = 3
MAX_PAGES_PER_PDF = 30
MAX_CHUNKS_PER_SOURCE = 300
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifacts", help="Output folder for chunks.jsonl / embeddings_cache.npz / clause_index.json")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--hash-local", action="store_true", help="Skip the Hugging Face model (hash embeddings)")
    parser.add_argument("--pdf-backend", default="auto", choices=["auto", "pymupdf", "pypdf"])
    parser.add_argument("--rebuild", action="store_true", help="Ignore the cached chunks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    artifacts = Path(args.artifacts)
    build = artifacts / "build"
    config = AppConfig(
        data_dir=build, raw_dir=build / "raw", processed_dir=build / "processed", indices_dir=build / "indices",
        manifest_path=build / "pdfs_manifest.json", clause_index_path=build / "clause_index.json",
        chunks_path=build / "processed" / "chunks.jsonl", embedding_model=args.embedding_model,
    )
    result = build_knowledge_base_from_sources(
        config=config, source_urls=DEFAULT_SOURCE_CATALOG, limit=SOURCE_LIMIT,
        max_pages_per_pdf=MAX_PAGES_PER_PDF, max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
        pdf_backend=args.pdf_backend, rebuild=args.rebuild, build_artifacts=False,
    )
    if result.failed_sources or result.failed_pdfs:
        print("Failed sources:", result.failed_sources, "Failed PDFs:", result.failed_pdfs, file=sys.stderr)
    if not result.chunks:
        print("No chunks were built; check internet access and the download log.", file=sys.stderr)
        return 1
    print("Chunks per source:", dict(Counter(str(c.get("source")) for c in result.chunks)))

    provider = None
    if not args.hash_local:
        provider = create_embedding_provider("huggingface", model_name=args.embedding_model, device="cpu")
    pipeline = RAGPipeline.from_chunks(result.chunks, embedding_provider=provider)
    print("Saved:", pipeline.save_artifacts(artifacts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
