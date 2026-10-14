from multimodal_rag.chunking import chunk_text


def test_chunks_keep_document_and_page_provenance():
    chunks = chunk_text(
        [
            {"page": 2, "text": "First page text."},
            {"page": 3, "text": "Second page text."},
        ],
        document_id="doc-123",
    )

    assert [item["chunk_index"] for item in chunks] == [0, 1]
    assert [item["page"] for item in chunks] == [2, 3]
    assert all(item["document_id"] == "doc-123" for item in chunks)
    assert all(item["source_type"] == "text" for item in chunks)
    assert [item["page_chunk_index"] for item in chunks] == [0, 0]


def test_empty_pages_do_not_create_chunks():
    chunks = chunk_text(
        [{"page": 1, "text": "  "}, {"page": 2, "text": "Useful text."}],
        document_id="doc-123",
    )

    assert len(chunks) == 1
    assert chunks[0]["page"] == 2
    assert chunks[0]["chunk_index"] == 0
