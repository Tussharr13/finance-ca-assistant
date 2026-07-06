from pathlib import Path

from finance_ca_assistant.evaluation.golden_testset import GoldenQuestion
from finance_ca_assistant.pipeline import RAGPipeline


def test_pipeline_answers_question_with_retrieval_trace(sample_chunks):
    pipeline = RAGPipeline.from_chunks(sample_chunks)

    response = pipeline.answer("What does Form 3CD clause 44 require?", top_k=2)

    assert response.retrieved[0].chunk["id"] == "tax-44"
    assert "clause" in response.retrieved[0].methods
    assert response.answer["basis"]["primary_sources"][0]["document"] == sample_chunks[0]["source"]


def test_pipeline_evaluates_golden_questions(sample_chunks):
    pipeline = RAGPipeline.from_chunks(sample_chunks)
    questions = [
        GoldenQuestion(
            id="q1",
            question="What does Form 3CD clause 44 require?",
            category="tax_audit",
            correct_sources=["ICAI_Tax_Audit_44AB_Guidance_Note_2023.pdf"],
            expected_sections=["form_3cd:44"],
            difficulty="easy",
        )
    ]

    report = pipeline.evaluate(questions, top_k=3)

    assert report.metrics["source_coverage"] == 1.0
    assert report.metrics["mrr"] == 1.0


def test_pipeline_saves_artifacts(tmp_path, sample_chunks):
    pipeline = RAGPipeline.from_chunks(sample_chunks)

    paths = pipeline.save_artifacts(tmp_path)

    assert Path(paths["chunks"]).exists()
    assert Path(paths["embeddings"]).exists()
    assert Path(paths["clause_index"]).exists()

