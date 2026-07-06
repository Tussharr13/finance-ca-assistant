import json

from finance_ca_assistant.ingestion.pdf_processor import PageText, ProcessedPDF
from finance_ca_assistant.knowledge_base import (
    chunks_from_processed_pdf,
    infer_document_type,
    write_chunks_jsonl,
)


def test_infer_document_type_for_known_sources():
    assert infer_document_type("ICAI_Tax_Audit_44AB_Guidance_Note_2023.pdf") == "form_3cd"
    assert infer_document_type("CGST_Rules_2017_CBIC.pdf") == "gst_rules"
    assert infer_document_type("ICAI_AS_22_Taxes_on_Income.pdf") == "accounting_standard"
    assert infer_document_type("ICAI_SA_500_Audit_Evidence.pdf") == "audit_standard"
    assert infer_document_type("unknown.pdf") == "generic"


def test_chunks_from_processed_pdf_adds_ids_metadata_and_references():
    processed = ProcessedPDF(
        source="ICAI_Tax_Audit_44AB_Guidance_Note_2023.pdf",
        path="/tmp/tax.pdf",
        pages=[
            PageText(
                page=221,
                text="Clause 44 of Form 3CD requires reporting expenditure relating to GST.",
            )
        ],
    )

    chunks = chunks_from_processed_pdf(processed)

    assert chunks[0]["id"] == "ICAI_Tax_Audit_44AB_Guidance_Note_2023.pdf::p221::c001"
    assert chunks[0]["document_type"] == "form_3cd"
    assert chunks[0]["references"] == ["form_3cd:44"]


def test_write_chunks_jsonl_round_trips(tmp_path):
    chunks = [{"id": "c1", "source": "x.pdf", "page": 1, "text": "AS 22 applies."}]
    output = tmp_path / "chunks.jsonl"

    write_chunks_jsonl(chunks, output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == chunks
