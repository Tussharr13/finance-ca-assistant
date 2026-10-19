"""Build the RAG pipeline for the server from environment settings.

FINANCE_CA_ARTIFACTS_DIR  Folder with chunks.jsonl / embeddings_cache.npz /
                          clause_index.json, e.g. your Kaggle artifacts dataset.
FINANCE_CA_EMBED_DEVICE   Device for a Hugging Face embedding model (default cpu).
FINANCE_CA_LLM_MODEL      Hugging Face LLM for answers, e.g. Qwen/Qwen3-4B. Unset
                          means no LLM: general questions abstain, tax
                          consultations still calculate.

With no artifacts the server runs calculator-only on an empty corpus.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from finance_ca_assistant.embeddings.embedding_cache import EmbeddingCache
from finance_ca_assistant.embeddings.embedding_factory import create_embedding_provider
from finance_ca_assistant.knowledge_base import load_chunks_jsonl
from finance_ca_assistant.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedPipeline:
    pipeline: RAGPipeline
    corpus: str  # human-readable description of what was loaded
    llm_model: Optional[str]


def load_pipeline_from_env() -> LoadedPipeline:
    llm_model = os.environ.get("FINANCE_CA_LLM_MODEL") or None
    llm = None
    if llm_model:
        from finance_ca_assistant.generation.llm_factory import create_llm

        llm = create_llm("huggingface", model_name=llm_model)

    artifacts = os.environ.get("FINANCE_CA_ARTIFACTS_DIR")
    if not artifacts:
        logger.warning("FINANCE_CA_ARTIFACTS_DIR not set: running calculator-only with no knowledge base")
        return LoadedPipeline(RAGPipeline.from_chunks([], llm=llm), "none", llm_model)

    root = Path(artifacts)
    chunks_path = root / "chunks.jsonl"
    embeddings_path = root / "embeddings_cache.npz"
    if not embeddings_path.exists():
        chunks = load_chunks_jsonl(chunks_path)
        pipeline = RAGPipeline.from_chunks(chunks, llm=llm)
        return LoadedPipeline(pipeline, f"{len(chunks)} chunks from {root} (hash-local embeddings)", llm_model)

    saved_model = EmbeddingCache(embeddings_path).load().model_name
    model_name = _resolve_embedding_model(saved_model, root)
    provider = None
    if model_name != "hash-local":
        provider = create_embedding_provider(
            "huggingface", model_name=model_name, device=os.environ.get("FINANCE_CA_EMBED_DEVICE", "cpu")
        )
    try:
        pipeline = RAGPipeline.from_artifacts(root, embedding_provider=provider, llm=llm)
        detail = "saved embeddings"
    except ValueError as error:
        # Usually the saved model name is a Kaggle path; re-embed with the local copy.
        logger.warning("Saved embeddings not reusable (%s); re-embedding chunks", error)
        pipeline = RAGPipeline.from_chunks(load_chunks_jsonl(chunks_path), embedding_provider=provider, llm=llm)
        detail = "re-embedded"
    return LoadedPipeline(pipeline, f"{len(pipeline.chunks)} chunks from {root} ({model_name}, {detail})", llm_model)


def _resolve_embedding_model(saved_model: str, root: Path) -> str:
    """Use the saved model if it is loadable here, else the adopted fine-tuned copy in ``root``."""

    if saved_model == "hash-local" or not saved_model.startswith("/") or Path(saved_model).exists():
        return saved_model
    from finance_ca_assistant.finetuning import find_finetuned_model

    finetuned = find_finetuned_model([root])
    return str(finetuned) if finetuned else "BAAI/bge-small-en-v1.5"
