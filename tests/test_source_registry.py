import json

from finance_ca_assistant.ingestion.source_registry import SourceRegistry


def test_source_registry_persists_legal_metadata(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"source bytes")
    manifest_path = tmp_path / "sources.json"

    SourceRegistry(manifest_path).register_source(
        "source",
        pdf_path,
        version="2026-1",
        metadata={
            "title": "Example Rules",
            "url": "https://example.test/rules.pdf",
            "authority": "Example Authority",
            "document_type": "rules",
            "jurisdiction": "India",
            "publication_date": "2026-04-01",
            "effective_from": "2026-04-01",
            "financial_year": "2026-27",
        },
    )

    record = json.loads(manifest_path.read_text(encoding="utf-8"))["source"]
    assert record["authority"] == "Example Authority"
    assert record["document_type"] == "rules"
    assert record["publication_date"] == "2026-04-01"
    assert record["effective_from"] == "2026-04-01"
    assert record["financial_year"] == "2026-27"
