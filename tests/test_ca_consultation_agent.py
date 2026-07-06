from finance_ca_assistant.pipeline import RAGPipeline


def test_salary_tax_agent_asks_for_missing_client_facts(sample_chunks):
    pipeline = RAGPipeline.from_chunks(sample_chunks)

    response = pipeline.consult(
        "I am earning salary and want to save tax legally. Act like my CA.",
        top_k=2,
    )

    assert response.status == "needs_info"
    assert "I need facts" in response.message
    assert response.questions
    assert any("annual CTC" in question for question in response.questions)
    assert response.retrieved_sources


def test_salary_tax_agent_uses_profile_updates_from_message(sample_chunks):
    pipeline = RAGPipeline.from_chunks(sample_chunks)

    response = pipeline.consult(
        "My salary is 20 LPA and I want tax saving legally. Compare both regimes.",
        top_k=2,
    )

    assert response.status == "needs_info"
    assert response.profile_updates["annual_salary_ctc"] == 2000000.0
    assert response.profile_updates["tax_regime_preference"] == "compare"
