import json

import pytest

from finance_ca_assistant.generation.ca_answer_generator import CAAnswerGenerator
from finance_ca_assistant.generation.llm_factory import LLMClient


class StubLLM(LLMClient):
    def __init__(self, payload):
        self.payload = payload

    def generate(self, *args, **kwargs):
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


def claim(chunk):
    return {"text": chunk["text"], "quote": chunk["text"], "chunk_id": chunk["id"]}


def test_valid_claim_does_not_render_unchecked_model_fields(sample_chunks):
    payload = {"claims": [claim(sample_chunks[0])], "answer": "UNSAFE", "risks": ["UNSAFE"], "ca_analysis": "UNSAFE", "basis": {"confidence": "high"}}
    result = CAAnswerGenerator(StubLLM(payload)).generate("question", sample_chunks)
    assert result["status"] == "answered"
    assert result["basis"]["evidence_status"] == "quotes_verified"
    assert result["basis"]["confidence"] == "not_assessed"
    assert result["basis"]["primary_sources"][0]["page"] == 221
    assert result["basis"]["primary_sources"][0]["quote"] == sample_chunks[0]["text"]
    assert "UNSAFE" not in json.dumps(result)


@pytest.mark.parametrize("payload", ["freeform advice", {}, {"claims": []}, {"claims": "wrong"}, {"claims": [None]}, {"claims": [{}]}])
def test_malformed_outputs_abstain(payload, sample_chunks):
    result = CAAnswerGenerator(StubLLM(payload)).generate("question", sample_chunks)
    assert result["status"] == "abstained"
    assert result["basis"]["primary_sources"] == []


@pytest.mark.parametrize("change", [
    {"text": "Clause 44 does not require reporting."},
    {"chunk_id": "invented"},
    {"quote": "Changed number 999"},
    {"text": 123},
])
def test_one_invalid_claim_blocks_every_claim(change, sample_chunks):
    bad = dict(claim(sample_chunks[0]), **change)
    result = CAAnswerGenerator(StubLLM({"claims": [claim(sample_chunks[1]), bad]})).generate("question", sample_chunks)
    assert result["status"] == "abstained"
    assert result["claims"] == []
    assert result["basis"]["primary_sources"] == []


@pytest.mark.parametrize("page", [None, 0, True, "221"])
def test_invalid_page_blocks(page, sample_chunks):
    chunk = dict(sample_chunks[0], page=page)
    result = CAAnswerGenerator(StubLLM({"claims": [claim(chunk)]})).generate("question", [chunk])
    assert result["status"] == "abstained"


def test_duplicate_ids_do_not_improve_confidence(sample_chunks):
    generator = CAAnswerGenerator(StubLLM({"claims": [claim(sample_chunks[0])]}))
    result = generator.generate("question", sample_chunks * 3)
    assert result["status"] == "abstained"
    assert result["basis"]["confidence"] == "not_assessed"


def test_cropped_negation_is_rejected():
    chunk = {"id": "a", "source": "law", "page": 1, "text": "No deduction is allowed."}
    cropped = {"text": "deduction is allowed.", "quote": "deduction is allowed.", "chunk_id": "a"}
    result = CAAnswerGenerator(StubLLM({"claims": [cropped]})).generate("question", [chunk])
    assert result["status"] == "abstained"


def test_echo_and_empty_context_never_claim_verified_evidence(sample_chunks):
    for chunks in (sample_chunks, []):
        result = CAAnswerGenerator().generate("question", chunks)
        assert result["status"] == "abstained"
        assert result["basis"]["confidence"] == "not_assessed"
        assert result["basis"]["primary_sources"] == []


LAW = {
    "id": "law-1",
    "source": "Income_Tax_Act.pdf",
    "page": 12,
    "text": (
        "The deduction under this section is allowed to an individual. "
        "No deduction shall be allowed for any premium paid in cash. "
        "The aggregate amount shall not exceed 25000 rupees, except where the insured is a senior citizen. "
        "Payment must be made in the previous year."
    ),
}


def paraphrase(text, quote):
    return {"text": text, "quote": quote, "chunk_id": LAW["id"]}


def test_paraphrase_backed_by_complete_sentences_is_answered():
    claims = [
        paraphrase(
            "A premium paid in cash gets no deduction.",
            "No deduction shall be allowed for any premium paid in cash.",
        ),
        paraphrase(
            "The total amount cannot exceed 25000 rupees, except for a senior citizen insured.",
            "The aggregate amount shall not exceed 25000 rupees, except where the insured is a senior citizen.",
        ),
    ]
    result = CAAnswerGenerator(StubLLM({"claims": claims})).generate("question", [LAW])
    assert result["status"] == "answered", result["diagnostics"]
    assert "[1] A premium paid in cash gets no deduction." in result["answer"]
    assert 'Source: Income_Tax_Act.pdf, page 12: "No deduction shall be allowed' in result["answer"]
    assert [source["page"] for source in result["basis"]["primary_sources"]] == [12, 12]


def test_model_json_inside_prose_or_fences_is_read():
    payload = '```json\n{"claims": [{"text": "A premium paid in cash gets no deduction.", ' \
        '"quote": "No deduction shall be allowed for any premium paid in cash.", "chunk_id": "law-1"}]}\n```'
    assert CAAnswerGenerator(StubLLM(payload)).generate("question", [LAW])["status"] == "answered"


@pytest.mark.parametrize(
    "claim, reason",
    [
        # Flipped meaning: the quote says "No deduction".
        (paraphrase("A premium paid in cash gets a deduction.", "No deduction shall be allowed for any premium paid in cash."),
         "negation_mismatch"),
        # Dropped the senior-citizen exception.
        (paraphrase("The total amount cannot exceed 25000 rupees.",
                    "The aggregate amount shall not exceed 25000 rupees, except where the insured is a senior citizen."),
         "dropped_qualifier"),
        # Changed figure.
        (paraphrase("The total amount cannot exceed 50000 rupees, except for senior citizens.",
                    "The aggregate amount shall not exceed 25000 rupees, except where the insured is a senior citizen."),
         "number_not_in_quote"),
        # Quote starts mid-sentence, cropping the "No".
        (paraphrase("Deduction shall be allowed for premium paid in cash.", "deduction shall be allowed for any premium paid in cash."),
         "quote_not_complete_sentences_from_chunk"),
        # Quote not in the chunk.
        (paraphrase("Premiums are always deductible.", "Premiums are always deductible."),
         "quote_not_complete_sentences_from_chunk"),
        # Claim unrelated to its quote.
        (paraphrase("Home loan principal qualifies for a separate benefit.", "Payment must be made in the previous year."),
         "insufficient_overlap_with_quote"),
    ],
)
def test_unfaithful_claims_abstain(claim, reason):
    result = CAAnswerGenerator(StubLLM({"claims": [claim]})).generate("question", [LAW])
    assert result["status"] == "abstained"
    assert result["diagnostics"]["citation_errors"] == [f"claim_0:{reason}"]


def test_quotes_longer_than_three_sentences_are_rejected():
    sentences = LAW["text"]
    claim = paraphrase("The deduction is allowed to an individual, not for cash premiums, up to 25000 rupees except "
                       "senior citizens, paid in the previous year.", sentences)
    result = CAAnswerGenerator(StubLLM({"claims": [claim]})).generate("question", [LAW])
    assert result["diagnostics"]["citation_errors"] == ["claim_0:quote_not_complete_sentences_from_chunk"]
