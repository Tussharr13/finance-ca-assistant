from finance_ca_assistant.generation.citation_validator import CitationValidator
from finance_ca_assistant.generation.hallucination_detector import HallucinationDetector


def test_hallucination_detector_flags_unknown_reference(sample_chunks):
    detector = HallucinationDetector.from_chunks(sample_chunks)

    report = detector.detect("Refer Form 3CD clause 44 and Section 999.")

    assert report.valid is False
    assert "income_tax_section:999" in report.invalid_references


def test_citation_validator_accepts_supported_claim(sample_chunks):
    validator = CitationValidator(min_overlap=0.5)

    result = validator.validate_claim("Clause 44 requires GST expenditure reporting", sample_chunks[0])

    assert result.valid is True

