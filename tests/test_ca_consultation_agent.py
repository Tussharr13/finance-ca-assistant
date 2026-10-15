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
    assert any("gross salary" in question for question in response.questions)
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

    first = pipeline.consult("My salary is 20 LPA, basic 10 lakh, age 30, resident. I want to save tax legally.", top_k=2)
    assert first.status == "needs_info"

    profile = dict(first.profile_updates)
    second = pipeline.consult(
        "Compare both regimes. I live in Mumbai, rent 30000, HRA 33,333 per month. "
        "EPF and PPF, 80C 1.5 lakh, health insurance 25000, no other income, have Form 16. Save tax.",
        profile=profile,
        top_k=2,
    )

    assert second.status == "calculated", second.message
    assert second.profile_updates["city"] == "mumbai"
    assert second.profile_updates["other_income"] == "none"
    assert second.calculation is not None

def test_amount_units_and_periods_are_normalized():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    assert _infer_profile_updates("my salary is 20L")["annual_salary_ctc"] == 2000000.0
    assert _infer_profile_updates("my monthly salary is 1,00,000")["annual_salary_ctc"] == 1200000.0
    assert _infer_profile_updates("salary 1 lakh per month")["annual_salary_ctc"] == 1200000.0
    assert _infer_profile_updates("rent is 3.6 lakh per year")["monthly_rent"] == 30000.0
    updates = _infer_profile_updates("salary 20 lpa rent 30000")
    assert updates["annual_salary_ctc"] == 2000000.0
    assert updates["monthly_rent"] == 30000.0


def test_amounts_do_not_cross_clauses_or_come_from_other_income():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    updates = _infer_profile_updates("I earn a salary, 32 years old")
    assert "annual_salary_ctc" not in updates
    assert updates["age"] == 32
    assert "annual_salary_ctc" not in _infer_profile_updates("freelance income 3 lakh")
    assert _infer_profile_updates("Age: 30")["age"] == 30


def test_negations_are_respected():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    assert _infer_profile_updates("I don't have health insurance")["health_insurance"] == "none"
    assert _infer_profile_updates("I don't have form 16 yet")["employer_tds_or_form16"] == "not available"
    assert _infer_profile_updates("I am not a resident")["residential_status"] == "non-resident"
    assert _infer_profile_updates("I am not paying rent")["monthly_rent"] == 0.0


def test_home_loan_interest_is_not_other_income():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    updates = _infer_profile_updates("I pay home loan interest")
    assert "other_income" not in updates
    assert updates["deductions_investments"] == "home loan"
    assert _infer_profile_updates("interest on fd 20000")["other_income"] == "interest"


def test_city_capture_stops_at_the_city_name():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    assert _infer_profile_updates("I live in bangalore with my wife")["city"] == "bangalore"
    assert "city" not in _infer_profile_updates("I live in a rented flat, salary 20 lpa")


def test_hra_is_stored_monthly_like_rent():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    updates = _infer_profile_updates("salary 18 lpa, rent 25k per month, HRA 7.2 lakh per year")
    assert updates["monthly_rent"] == 25000.0
    assert updates["monthly_hra"] == 60000.0
    assert _infer_profile_updates("HRA 20,000")["monthly_hra"] == 20000.0
    assert _infer_profile_updates("I don't get HRA")["monthly_hra"] == 0.0


def _complete_profile(**overrides):
    profile = {
        "annual_salary_ctc": 2_000_000.0,
        "annual_basic_da": 1_000_000.0,
        "age": 30,
        "residential_status": "resident",
        "monthly_rent": 30_000.0,
        "monthly_hra": 400_000.0 / 12,
        "city": "mumbai",
        "deductions_investments": "EPF, PPF",
        "deduction_80c_amount": 150_000.0,
        "health_insurance": "yes",
        "health_premium_self_family": 25_000.0,
        "other_income": "none",
        "other_income_amount": 0.0,
        "professional_tax": 2_500.0,
    }
    profile.update(overrides)
    return profile


def test_calculation_turn_matches_the_calculator(sample_chunks):
    from finance_ca_assistant.calculators import SalaryTaxInput, compare_regimes

    response = RAGPipeline.from_chunks(sample_chunks).consult("Help me save tax", profile=_complete_profile(), top_k=2)

    expected = compare_regimes(
        SalaryTaxInput(
            tax_year="FY2025-26", gross_salary=2_000_000, basic_da=1_000_000, age=30,
            hra_received=400_000, rent_paid=360_000, metro=True, professional_tax=2_500,
            deduction_80c=150_000, health_premium_self_family=25_000,
        )
    )
    assert response.status == "calculated"
    assert response.calculation.run_id == expected.run_id
    assert response.calculation.old.total_tax == 276_900
    assert response.calculation.new.total_tax == 192_400
    assert "\u20b92,76,900" in response.message and "\u20b91,92,400" in response.message
    assert "The new regime is lower by \u20b984,500" in response.message
    assert "Section 10(13A)" in response.message
    assert "Not included in this calculation" in response.message
    assert response.answer is None


def test_amount_questions_are_asked_only_for_items_the_client_has(sample_chunks):
    pipeline = RAGPipeline.from_chunks(sample_chunks)
    profile = _complete_profile(
        deduction_80c_amount=None, health_premium_self_family=None,
        deductions_investments="NPS, home loan", other_income="interest", other_income_amount=None,
    )
    response = pipeline.consult("save tax", profile=profile, top_k=2)
    asked = " ".join(response.questions)
    assert response.status == "needs_info"
    for phrase in ("Section 80C", "NPS", "home-loan interest", "health insurance premium", "other income"):
        assert phrase in asked

    no_rent = _complete_profile(monthly_rent=0.0, monthly_hra=None, city=None)
    assert pipeline.consult("save tax", profile=no_rent, top_k=2).status == "calculated"


def test_basic_above_salary_is_asked_again(sample_chunks):
    response = RAGPipeline.from_chunks(sample_chunks).consult(
        "save tax", profile=_complete_profile(annual_basic_da=3_000_000.0), top_k=2
    )
    assert response.status == "needs_info"
    assert any("basic salary" in question for question in response.questions)


def test_unverified_tax_year_is_refused_politely(sample_chunks):
    response = RAGPipeline.from_chunks(sample_chunks).consult(
        "Plan my tax for FY 2026-27", profile=_complete_profile(), top_k=2
    )
    assert response.status == "unsupported_year"
    assert "FY2026-27" in response.message and "FY2025-26" in response.message


def test_calculator_amounts_are_parsed_from_chat():
    from finance_ca_assistant.agents.ca_consultation_agent import _infer_profile_updates

    updates = _infer_profile_updates(
        "basic salary 50,000 per month, 80C 1.5 lakh, NPS 50000, parents insurance 30000, "
        "health insurance 20000, senior citizen parents, home loan interest 1.8 lakh, interest income 40000"
    )
    assert updates["annual_basic_da"] == 600_000.0
    assert updates["deduction_80c_amount"] == 150_000.0
    assert updates["nps_self_amount"] == 50_000.0
    assert updates["health_premium_parents"] == 30_000.0
    assert updates["health_premium_self_family"] == 20_000.0
    assert updates["parents_senior"] is True
    assert updates["home_loan_interest"] == 180_000.0
    assert updates["other_income_amount"] == 40_000.0

    none_answers = _infer_profile_updates("no investments, no health insurance, no other income")
    assert none_answers["deduction_80c_amount"] == 0.0
    assert none_answers["health_premium_self_family"] == 0.0
    assert none_answers["other_income_amount"] == 0.0


def test_inr_uses_indian_grouping():
    from finance_ca_assistant.agents.ca_consultation_agent import _inr

    assert _inr(1_234_567) == "\u20b912,34,567"
    assert _inr(999) == "\u20b9999"
    assert _inr(100_000) == "\u20b91,00,000"


def test_follow_up_answers_continue_the_consultation(sample_chunks):
    pipeline = RAGPipeline.from_chunks(sample_chunks)
    first = pipeline.consult("My salary is 18 LPA, basic 9 lakh, age 34, resident. Help me save tax.", top_k=2)
    assert first.status == "needs_info"

    second = pipeline.consult(
        "I live in Pune, rent 25k per month, HRA 30000 per month. 80C 1.5 lakh via EPF and ELSS. "
        "health insurance 20000, parents insurance 30000, senior citizen parents. no other income",
        profile=first.profile_updates,
        top_k=2,
    )
    assert second.status == "calculated", second.message

    # A fresh general question with no facts is not forced into the tax flow.
    general = pipeline.consult("What does SA 500 say about audit evidence?", profile={"age": 34}, top_k=2)
    assert general.status != "needs_info"
