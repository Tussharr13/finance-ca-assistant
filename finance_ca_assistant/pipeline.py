"""End-to-end orchestration for the Finance CA RAG system."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from finance_ca_assistant.embeddings.embedding_cache import EmbeddingCache
from finance_ca_assistant.embeddings.embedding_factory import (
    EmbeddingProvider,
    create_embedding_provider,
)
from finance_ca_assistant.evaluation.evaluator import EvaluationReport, EvaluationRunner
from finance_ca_assistant.evaluation.golden_testset import GoldenQuestion
from finance_ca_assistant.generation.ca_answer_generator import CAAnswerGenerator
from finance_ca_assistant.generation.llm_factory import LLMClient
from finance_ca_assistant.processing.clause_extractor import ClauseExtractor
from finance_ca_assistant.processing.metadata_builder import MetadataBuilder
from finance_ca_assistant.retrieval.bm25_index import BM25Index
from finance_ca_assistant.retrieval.clause_searcher import ClauseSearcher
from finance_ca_assistant.retrieval.hybrid_retriever import HybridRetriever, HybridSearchResult
from finance_ca_assistant.retrieval.vector_store_base import InMemoryVectorStore


@dataclass(frozen=True)
class PipelineResponse:
    """Complete response from retrieval through answer generation."""

    question: str
    answer: Dict[str, Any]
    retrieved: List[HybridSearchResult]


class RAGPipeline:
    """Runnable end-to-end CA RAG pipeline.

    This class is the ergonomic entry point for notebooks, scripts, tests, and
    FastAPI. It wires together embedding, vector search, BM25, direct clause
    lookup, hybrid fusion, and structured answer generation.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedding_provider: EmbeddingProvider,
        embeddings: Any,
        retriever: HybridRetriever,
        answer_generator: CAAnswerGenerator,
    ) -> None:
        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self.embeddings = embeddings
        self.retriever = retriever
        self.answer_generator = answer_generator

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Dict[str, Any]],
        embedding_provider: EmbeddingProvider | None = None,
        reranker: object | None = None,
        llm: LLMClient | None = None,
    ) -> "RAGPipeline":
        """Build a complete local pipeline from chunk dictionaries."""

        metadata_builder = MetadataBuilder()
        enriched_chunks = [metadata_builder.enrich_chunk(dict(chunk)) for chunk in chunks]
        provider = embedding_provider or create_embedding_provider("hash-local", dimensions=384)
        embeddings = provider.embed_texts([str(chunk.get("text") or "") for chunk in enriched_chunks])

        vector_store = InMemoryVectorStore()
        vector_store.add(embeddings, enriched_chunks)
        bm25_index = BM25Index()
        bm25_index.add(enriched_chunks)
        clause_searcher = ClauseSearcher.from_chunks(enriched_chunks)
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            clause_searcher=clause_searcher,
            query_embedder=provider.embed_query,
            reranker=reranker,
        )
        return cls(
            chunks=enriched_chunks,
            embedding_provider=provider,
            embeddings=embeddings,
            retriever=retriever,
            answer_generator=CAAnswerGenerator(llm=llm),
        )

    @classmethod
    def from_chunks_jsonl(cls, path: str | Path) -> "RAGPipeline":
        """Load chunks from JSONL and build a pipeline."""

        chunk_path = Path(path)
        chunks = [
            json.loads(line)
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls.from_chunks(chunks)

    def answer(self, question: str, top_k: int = 6) -> PipelineResponse:
        """Retrieve relevant chunks and generate a structured CA answer."""

        retrieved = self.retriever.retrieve(question, top_k=top_k)
        answer = self.answer_generator.generate(question, [result.chunk for result in retrieved])
        return PipelineResponse(question=question, answer=answer, retrieved=retrieved)

    def consult(
        self,
        user_message: str,
        profile: Dict[str, Any] | None = None,
        history: List[Dict[str, str]] | None = None,
        top_k: int = 6,
    ) -> object:
        """Run an agentic CA consultation turn."""

        from finance_ca_assistant.agents.ca_consultation_agent import CAConsultationAgent

        return CAConsultationAgent(self).respond(
            user_message=user_message,
            profile=profile,
            history=history,
            top_k=top_k,
        )

    def retrieve_chunks(self, question: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Return only retrieved chunk dictionaries for evaluation/API adapters."""

        return [result.chunk for result in self.retriever.retrieve(question, top_k=top_k)]

    def evaluate(self, questions: List[GoldenQuestion], top_k: int = 10) -> EvaluationReport:
        """Evaluate retrieval against golden questions."""

        runner = EvaluationRunner(questions)
        return runner.evaluate_retrieval(lambda question: self.retrieve_chunks(question, top_k=top_k))

    def save_artifacts(self, output_dir: str | Path) -> Dict[str, str]:
        """Save chunks, embeddings, and clause index artifacts."""

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        chunks_path = root / "chunks.jsonl"
        embeddings_path = root / "embeddings_cache.npz"
        clause_index_path = root / "clause_index.json"

        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk, sort_keys=True) + "\n")

        EmbeddingCache(embeddings_path).save(
            self.embeddings,
            self.chunks,
            model_name=self.embedding_provider.model_name,
        )
        clause_index = ClauseExtractor().build_index(self.chunks)
        clause_index_path.write_text(
            json.dumps(clause_index, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "chunks": str(chunks_path),
            "embeddings": str(embeddings_path),
            "clause_index": str(clause_index_path),
        }
