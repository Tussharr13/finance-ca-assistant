import json

import pytest

from finance_ca_assistant.calculators import SalaryTaxInput, compare_regimes
from finance_ca_assistant.generation.llm_factory import LLMClient
from finance_ca_assistant.generation.typed_claims import (
    build_claims_prompt,
    build_salary_evidence,
    extract_numbers,
    render_claims,
    validate_claims,
)
from finance_ca_assistant.pipeline import RAGPipeline

PROFILE = {
    "annual_salary_ctc": 1_800_000.0,
    "annual_basic_da": 900_000.0,
    "age": 34,
    "residential_status": "resident",
    "city": "pune",
    "monthly_rent": 25_000.0,
    "monthly_hra": 30_000.0,
    "deductions_investments": "EPF, ELSS",
    "deduction_80c_amount": 150_000.0,
    "health_insurance": "yes",
    "health_premium_self_family": 20_000.0,
    "health_premium_parents": 30_000.0,
    "parents_senior": True,
    "other_income": "none",
    "other_income_amount": 0.0,
}
CHUNK = {
    "id": "it-80c-p4",
    "source": "Income_Tax_Act_1961.pdf",
    "page": 4,
    "text": "The aggregate deduction under section 80C shall not exceed one hundred and fifty thousand rupees (150000).",
}


@pytest.fixture(scope="module")
def comparison():
    return compare_regimes(
        SalaryTaxInput(
            tax_year="FY2025-26", gross_salary=1_800_000, basic_da=900_000, age=34,
            hra_received=360_000, rent_paid=300_000, deduction_80c=150_000,
            health_premium_self_family=20_000, health_premium_parents=30_000, parents_senior=True,
        )
    )


@pytest.fixture(scope="module")
def evidence(comparison):
    return build_salary_evidence(PROFILE, comparison, [CHUNK])


def claims(*items):
    return json.dumps({"claims": list(items)})


def test_valid_claims_are_accepted_and_rendered(comparison, evidence):
    raw = claims(
        {"type": "fact", "text": "Your gross salary is ₹18,00,000 a year.", "refs": ["fact:annual_salary_ctc"]},
        {"type": "calculation", "text": "Tax is ₹2,23,080 under the old regime and ₹1,50,800 under the new.",
         "refs": ["calc:old.total_tax", "calc:new.total_tax"]},
        {"type": "calculation", "text": "Your HRA exemption of ₹2,10,000 applies only in the old regime.",
         "refs": ["calc:old.step1"]},
        {"type": "rule", "text": "Section 80C deductions are capped at 150000 rupees.", "refs": ["chunk:it-80c-p4"]},
        {"type": "recommendation", "text": "With ₹3 lakh rent a year, the new regime still saves you about ₹0.72 lakh.",
         "refs": ["fact:monthly_rent", "calc:saving"]},
    )
    result = validate_claims(raw, evidence, lower_regime=comparison.lower_regime)
    assert result.rejected == [] and len(result.accepted) == 5
    rendered = render_claims(result.accepted, evidence)
    assert "(Section 10(13A) read with Rule 2A)" in rendered
    assert "(Income_Tax_Act_1961.pdf, page 4)" in rendered


@pytest.mark.parametrize(
    "claim, reason",
    [
        ({"type": "calculation", "text": "New regime tax is ₹1,40,800.", "refs": ["calc:new.total_tax"]},
         "unsupported_number:140800"),
        ({"type": "calculation", "text": "You save ₹72,000.", "refs": ["calc:saving"]}, "unsupported_number:72000"),
        ({"type": "calculation", "text": "Tax is ₹1,50,800.", "refs": ["calc:new.made_up"]},
         "unknown_ref:calc:new.made_up"),
        ({"type": "fact", "text": "You pay ₹1,50,800 tax.", "refs": ["calc:new.total_tax"]}, "fact_must_cite_only_facts"),
        ({"type": "rule", "text": "Rent above 1 lakh needs landlord PAN.", "refs": ["chunk:it-80c-p4"]},
         "unsupported_number:100000"),
        ({"type": "recommendation", "text": "Choose the old regime.", "refs": ["fact:age", "calc:old.total_tax"]},
         "contradicts_calculation"),
        ({"type": "recommendation", "text": "Switch to the new regime.", "refs": ["calc:saving"]},
         "recommendation_needs_fact_and_support"),
        ({"type": "opinion", "text": "Taxes are high.", "refs": ["fact:age"]}, "unknown_type"),
        ({"type": "calculation", "text": "The 30% slab applies to you.", "refs": ["calc:old.total_tax"]},
         "unsupported_number:30"),
    ],
)
def test_invalid_claims_are_rejected_with_a_reason(comparison, evidence, claim, reason):
    result = validate_claims(claims(claim), evidence, lower_regime=comparison.lower_regime)
    assert result.accepted == []
    assert result.rejected[0]["reason"] == reason


def test_one_bad_claim_does_not_drop_the_good_ones(comparison, evidence):
    raw = claims(
        {"type": "calculation", "text": "New regime tax is ₹1,50,800.", "refs": ["calc:new.total_tax"]},
        {"type": "calculation", "text": "Old regime tax is ₹2,00,000.", "refs": ["calc:old.total_tax"]},
    )
    result = validate_claims(raw, evidence, lower_regime="new")
    assert [claim["text"] for claim in result.accepted] == ["New regime tax is ₹1,50,800."]
    assert len(result.rejected) == 1


def test_malformed_output_is_an_error(evidence):
    assert validate_claims("I think the new regime is better.", evidence).error == "invalid_json_or_no_claims"
    assert validate_claims('{"claims": []}', evidence).error == "invalid_json_or_no_claims"


def test_number_extraction_ignores_legal_references_and_years():
    assert extract_numbers("Section 80C and Section 10(13A) for FY 2025-26, Form 16, calc_ab12") == []
    assert extract_numbers("Rs.72,280") == [(72_280.0, 1.0)]
    assert extract_numbers("₹2 lakh") == [(200_000.0, 10_000.0)]


def test_prompt_lists_evidence_ids_and_values(evidence):
    payload = json.loads(build_claims_prompt("Which regime?", evidence).removesuffix("/no_think"))
    ids = {row["id"]: row for row in payload["evidence"]}
    assert ids["calc:saving"]["value"] == 72_280.0
    assert "fact:annual_salary_ctc" in ids and "chunk:it-80c-p4" in ids


class ScriptedLLM(LLMClient):
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, system_prompt, user_prompt, max_tokens=900):
        self.prompts.append(user_prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _consult(llm):
    pipeline = RAGPipeline.from_chunks([CHUNK], llm=llm)
    return pipeline.consult("Which regime should I pick to save tax?", profile=PROFILE, top_k=1)


def test_consultation_shows_only_verified_explanation():
    llm = ScriptedLLM(claims(
        {"type": "calculation", "text": "The new regime saves you ₹72,280.", "refs": ["calc:saving"]},
        {"type": "calculation", "text": "The new regime saves you ₹90,000.", "refs": ["calc:saving"]},
    ))
    response = _consult(llm)
    assert response.status == "calculated"
    assert response.message.startswith("Why this result:\n- The new regime saves you ₹72,280.")
    assert "₹90,000" not in response.message
    assert len(response.explanation["rejected"]) == 1
    assert "| Total tax |" in response.message


def test_consultation_keeps_the_calculation_when_the_model_fails():
    for llm in (ScriptedLLM("not json"), ScriptedLLM(RuntimeError("CUDA out of memory"))):
        response = _consult(llm)
        assert response.status == "calculated"
        assert not response.message.startswith("Why this result")
        assert "| Total tax |" in response.message
        assert response.explanation["error"]


def test_no_explanation_without_a_real_llm():
    response = _consult(None)
    assert response.status == "calculated" and response.explanation is None
