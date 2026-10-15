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


def test_valid_extract_does_not_render_unchecked_model_fields(sample_chunks):
    payload = {"claims": [claim(sample_chunks[0])], "answer": "UNSAFE", "risks": ["UNSAFE"], "ca_analysis": "UNSAFE", "basis": {"confidence": "high"}}
    result = CAAnswerGenerator(StubLLM(payload)).generate("question", sample_chunks)
    assert result["status"] == "evidence_only"
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
