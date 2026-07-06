import numpy as np

from finance_ca_assistant.embeddings.embedding_factory import (
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
)
from finance_ca_assistant.generation.llm_factory import HuggingFacePipelineLLM, create_llm
from finance_ca_assistant.pipeline import RAGPipeline
from finance_ca_assistant.retrieval.reranker_factory import CrossEncoderReranker, create_reranker
from finance_ca_assistant.retrieval.hybrid_retriever import HybridSearchResult


class FakeSentenceTransformer:
    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        assert batch_size == 2
        assert normalize_embeddings is True
        rows = []
        for index, _text in enumerate(texts, start=1):
            rows.append([float(index), 0.0, 0.0])
        return np.array(rows, dtype=np.float32)


def test_sentence_transformer_embedding_provider_uses_injected_model():
    provider = SentenceTransformerEmbeddingProvider(
        model=FakeSentenceTransformer(),
        model_name="BAAI/bge-m3",
        batch_size=2,
        dimensions=3,
    )

    embeddings = provider.embed_texts(["first", "second"])

    assert provider.model_name == "BAAI/bge-m3"
    assert provider.dimensions == 3
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (2, 3)


def test_embedding_factory_creates_bge_m3_provider_with_injected_model():
    provider = create_embedding_provider("bge-m3", model=FakeSentenceTransformer(), dimensions=3)

    assert isinstance(provider, SentenceTransformerEmbeddingProvider)
    assert provider.model_name == "BAAI/bge-m3"


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.1 if "weak" in pair[1] else 0.9 for pair in pairs]


def test_cross_encoder_reranker_prioritizes_model_relevance():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder(), model_name="BAAI/bge-reranker-v2-m3")
    results = [
        HybridSearchResult(chunk={"id": "weak", "text": "weak match"}, score=0.95, methods=["dense"]),
        HybridSearchResult(chunk={"id": "strong", "text": "strong statutory answer"}, score=0.20, methods=["bm25"]),
    ]

    ranked = reranker.rerank("statutory answer", results, top_k=1)

    assert ranked[0].chunk["id"] == "strong"
    assert ranked[0].methods == ["bm25", "rerank"]


def test_reranker_factory_creates_cross_encoder_with_injected_model():
    reranker = create_reranker("bge-reranker-v2-m3", model=FakeCrossEncoder())

    assert isinstance(reranker, CrossEncoderReranker)


class FakePipeline:
    def __init__(self, expected_max_tokens=900):
        self.expected_max_tokens = expected_max_tokens

    def __call__(self, messages, max_new_tokens=900, temperature=0.1, do_sample=False):
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert max_new_tokens == self.expected_max_tokens
        return [{"generated_text": [{"role": "assistant", "content": "grounded answer"}]}]


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        return np.zeros((1, 11), dtype=np.int64)


class FakePipelineWithTokenizer:
    tokenizer = FakeTokenizer()

    def __call__(self, messages, max_length=None, max_new_tokens=None, temperature=0.1, do_sample=False):
        assert max_length == 53
        assert max_new_tokens is None
        assert do_sample is False
        return [{"generated_text": [{"role": "assistant", "content": "grounded answer"}]}]


def test_huggingface_pipeline_llm_generates_from_chat_messages():
    llm = HuggingFacePipelineLLM(pipeline=FakePipeline(expected_max_tokens=42), model_name="Qwen/Qwen3-4B")

    answer = llm.generate("system", "user", max_tokens=42)

    assert llm.model_name == "Qwen/Qwen3-4B"
    assert answer == "grounded answer"


def test_huggingface_pipeline_llm_uses_max_length_when_tokenizer_available():
    llm = HuggingFacePipelineLLM(pipeline=FakePipelineWithTokenizer(), model_name="Qwen/Qwen3-4B")

    answer = llm.generate("system", "user", max_tokens=42)

    assert answer == "grounded answer"


def test_llm_factory_creates_qwen_provider_with_injected_pipeline():
    llm = create_llm("qwen3-4b", pipeline=FakePipeline())

    assert isinstance(llm, HuggingFacePipelineLLM)
    assert llm.model_name == "Qwen/Qwen3-4B"


def test_pipeline_accepts_hf_embedding_reranker_and_llm(sample_chunks):
    embedding_provider = create_embedding_provider(
        "bge-m3",
        model=FakeSentenceTransformer(),
        dimensions=3,
        batch_size=2,
    )
    reranker = create_reranker("bge-reranker-v2-m3", model=FakeCrossEncoder())
    llm = create_llm("qwen3-4b", pipeline=FakePipeline())

    pipeline = RAGPipeline.from_chunks(
        sample_chunks,
        embedding_provider=embedding_provider,
        reranker=reranker,
        llm=llm,
    )
    response = pipeline.answer("What does Form 3CD clause 44 require?", top_k=2)

    assert pipeline.embedding_provider.model_name == "BAAI/bge-m3"
    assert pipeline.answer_generator.llm.model_name == "Qwen/Qwen3-4B"
    assert "rerank" in response.retrieved[0].methods
