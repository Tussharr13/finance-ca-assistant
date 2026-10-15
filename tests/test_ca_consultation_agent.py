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


def test_non_resident_is_not_parsed_as_resident():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    assert _infer_profile_updates("I am a non-resident")["residential_status"] == "non-resident"
    assert _infer_profile_updates("I am an NRI")["residential_status"] == "non-resident"
    assert _infer_profile_updates("I am RNOR")["residential_status"] == "not-ordinarily-resident"
    assert _infer_profile_updates("I am resident in India")["residential_status"] == "resident"


def test_amounts_with_digit_grouping_are_parsed():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    assert _infer_profile_updates("my salary is 1,200,000")["annual_salary_ctc"] == 1200000.0
    assert _infer_profile_updates("salary 12,00,000")["annual_salary_ctc"] == 1200000.0
    assert _infer_profile_updates("I pay rent of 25k")["monthly_rent"] == 25000.0


def test_unrelated_numbers_are_not_read_as_salary():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    assert "annual_salary_ctc" not in _infer_profile_updates("I have salary slips and Form 16")
    assert "annual_salary_ctc" not in _infer_profile_updates("interest income 50000")
    assert "annual_salary_ctc" not in _infer_profile_updates("other income 5 lakh")


def test_all_salary_facts_can_be_gathered_from_chat(sample_chunks):
    pipeline = RAGPipeline.from_chunks(sample_chunks)

    first = pipeline.consult("My salary is 20 LPA, age 30, resident. I want to save tax legally.", top_k=2)
    assert first.status == "needs_info"

    profile = dict(first.profile_updates)
    second = pipeline.consult(
        "Compare both regimes. I live in Mumbai, rent 30000, HRA 20,000. "
        "EPF and PPF, health insurance for parents, no other income, have Form 16. Save tax.",
        profile=profile,
        top_k=2,
    )

    assert second.status != "needs_info"
    assert second.profile_updates["city"] == "mumbai"
    assert second.profile_updates["other_income"] == "none"
