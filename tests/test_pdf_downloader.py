from finance_ca_assistant.ingestion.pdf_downloader import PDFDownloader


class StreamingResponse:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    @property
    def content(self):
        raise AssertionError("streaming downloads must not access response.content")

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        assert chunk_size == 1024 * 1024
        yield from self.chunks

    def close(self):
        self.closed = True


class StreamingSession:
    def __init__(self, response):
        self.response = response

    def get(self, url, timeout, stream):
        assert stream is True
        return self.response


def test_pdf_downloader_streams_to_disk(tmp_path):
    response = StreamingResponse([b"%PDF-1.7\n", b"bounded content"])
    downloader = PDFDownloader(tmp_path, session=StreamingSession(response))

    result = downloader.download("https://example.com/test.pdf")

    assert result.path.read_bytes() == b"%PDF-1.7\nbounded content"
    assert result.from_cache is False
    assert response.closed is True
    assert not (tmp_path / "test.pdf.part").exists()


def test_pdf_downloader_prefers_existing_cache_without_network(tmp_path):
    cached_path = tmp_path / "cached.pdf"
    cached_path.write_bytes(b"%PDF-cached")

    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("network must not be called when cache is preferred")

    result = PDFDownloader(tmp_path, session=FailingSession()).download(
        "https://example.com/cached.pdf",
        prefer_cache=True,
    )

    assert result.path == cached_path
    assert result.from_cache is True
