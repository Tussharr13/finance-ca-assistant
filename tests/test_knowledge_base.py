from finance_ca_assistant.config import AppConfig
from finance_ca_assistant.ingestion.pdf_processor import PageText, ProcessedPDF
from finance_ca_assistant.knowledge_base import (
    build_chunks_from_pdf_paths,
    build_knowledge_base_from_sources,
    chunks_from_processed_pdf,
    infer_document_type,
    load_chunks_jsonl,
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


def test_chunks_from_processed_pdf_stops_at_limit():
    processed = ProcessedPDF(
        source="large.pdf",
        path="/tmp/large.pdf",
        pages=[PageText(page=1, text="One.\n\nTwo.\n\nThree.\n\nFour.")],
    )

    chunks = chunks_from_processed_pdf(processed, max_chunks=2)

    assert len(chunks) == 2


def test_write_chunks_jsonl_round_trips(tmp_path):
    chunks = [{"id": "c1", "source": "x.pdf", "page": 1, "text": "AS 22 applies."}]
    output = tmp_path / "chunks.jsonl"

    write_chunks_jsonl(chunks, output)

    assert load_chunks_jsonl(output) == chunks


def test_build_chunks_from_pdf_paths_caps_chunks_per_source(monkeypatch, tmp_path):
    processed = ProcessedPDF(
        source="sample.pdf",
        path=str(tmp_path / "sample.pdf"),
        pages=[
            PageText(page=1, text="First paragraph.\n\nSecond paragraph.\n\nThird paragraph."),
        ],
    )
    monkeypatch.setattr(
        "finance_ca_assistant.knowledge_base.PDFProcessor.process",
        lambda self, path, max_pages=None: processed,
    )

    chunks, failures = build_chunks_from_pdf_paths(
        [tmp_path / "sample.pdf"],
        max_chunks_per_source=2,
    )

    assert not failures
    assert len(chunks) == 2


def test_build_knowledge_base_reuses_cached_chunks_before_download(monkeypatch, tmp_path):
    chunks = [{"id": "cached", "source": "cached.pdf", "page": 1, "text": "Cached text"}]
    config = AppConfig(
        data_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        indices_dir=tmp_path / "indices",
        manifest_path=tmp_path / "manifest.json",
        clause_index_path=tmp_path / "clause_index.json",
        chunks_path=tmp_path / "processed/chunks.jsonl",
    )
    write_chunks_jsonl(chunks, config.chunks_path)

    def fail_download(*args, **kwargs):
        raise AssertionError("cached build must not download sources")

    monkeypatch.setattr(
        "finance_ca_assistant.knowledge_base.PDFDownloader.download",
        fail_download,
    )

    result = build_knowledge_base_from_sources(
        config=config,
        source_urls={"unused": "https://example.com/unused.pdf"},
        rebuild=False,
        build_artifacts=False,
    )

    assert result.reused_chunks is True
    assert result.chunks == chunks
    assert result.pdf_paths == []
