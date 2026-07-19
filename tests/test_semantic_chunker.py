from finance_ca_assistant.processing.semantic_chunker import SemanticChunker


def test_final_window_with_trailing_character_always_finishes():
    chunker = SemanticChunker(max_chars=1200, overlap=150)
    # A boundary one character before the end previously made the overlap
    # return to the same start offset indefinitely.
    text = ("A" * 1199) + "\nZ"

    chunks = chunker.chunk(text, document_type="gst_rules")

    assert len(chunks) == 2
    assert chunks[-1].text.endswith("Z")


def test_chunks_do_not_exceed_configured_size():
    chunker = SemanticChunker(max_chars=100, overlap=20)
    text = "Paragraph sentence. " * 30

    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    assert all(0 < len(chunk.text) <= 100 for chunk in chunks)
