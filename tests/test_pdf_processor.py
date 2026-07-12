import sys
from types import SimpleNamespace

from finance_ca_assistant.ingestion.pdf_processor import PDFProcessor


class FakeTools:
    def __init__(self):
        self.shrink_calls = []

    def store_shrink(self, percent):
        self.shrink_calls.append(percent)
        return 0


class FakePage:
    def __init__(self, text):
        self.text = text

    def get_text(self, output, sort=False):
        assert output == "text"
        assert sort is True
        return self.text


class FakeDocument:
    def __init__(self, page_texts, needs_pass=False):
        self.page_texts = page_texts
        self.needs_pass = needs_pass
        self.page_count = len(page_texts)
        self.closed = False

    def load_page(self, page_index):
        return FakePage(self.page_texts[page_index])

    def close(self):
        self.closed = True


def test_pymupdf_backend_bounds_pages_text_and_cache(monkeypatch, tmp_path):
    document = FakeDocument(["A" * 20, "second page", "third page"])
    tools = FakeTools()
    fake_pymupdf = SimpleNamespace(open=lambda path: document, TOOLS=tools)
    monkeypatch.setitem(sys.modules, "pymupdf", fake_pymupdf)

    result = PDFProcessor(
        backend="pymupdf",
        max_page_text_chars=12,
        cache_shrink_interval=1,
    ).process(tmp_path / "sample.pdf", max_pages=2)

    assert result.metadata["backend"] == "pymupdf"
    assert result.metadata["processed_page_count"] == 2
    assert [page.text for page in result.pages] == ["A" * 12, "second page"]
    assert result.errors == ["page_1_text_truncated"]
    assert document.closed is True
    assert tools.shrink_calls == [100, 100, 100]


def test_pymupdf_backend_closes_encrypted_document(monkeypatch, tmp_path):
    document = FakeDocument([], needs_pass=True)
    tools = FakeTools()
    fake_pymupdf = SimpleNamespace(open=lambda path: document, TOOLS=tools)
    monkeypatch.setitem(sys.modules, "pymupdf", fake_pymupdf)

    result = PDFProcessor(backend="pymupdf").process(tmp_path / "encrypted.pdf")

    assert result.errors == ["encrypted_pdf"]
    assert result.metadata["backend"] == "pymupdf"
    assert document.closed is True
    assert tools.shrink_calls == [100]
