from multimodal_rag import embedding


class FakeCollection:
    def upsert(self, **kwargs):
        self.payload = kwargs


class FakeVectorStore:
    def __init__(self, **kwargs):
        self._collection = FakeCollection()


def test_indexing_deduplicates_exact_same_page_chunks_and_keeps_metadata(monkeypatch):
    monkeypatch.setattr(embedding, "Chroma", FakeVectorStore)
    monkeypatch.setattr(
        type(embedding.embeddings),
        "embed_documents",
        lambda _self, texts: [[0.1] for _ in texts],
    )
    chunks = [
        {
            "text": "Same passage.",
            "document_id": "doc-1",
            "page": 2,
            "chunk_index": 4,
            "page_chunk_index": 1,
        },
        {
            "text": "Same passage.",
            "document_id": "doc-1",
            "page": 2,
            "chunk_index": 5,
            "page_chunk_index": 2,
        },
        {
            "text": "Different passage.",
            "document_id": "doc-1",
            "page": 2,
            "chunk_index": 6,
            "page_chunk_index": 3,
        },
    ]

    store = embedding.build_vectorstore(chunks)
    payload = store._collection.payload

    assert payload["documents"] == ["Same passage.", "Different passage."]
    assert len(set(payload["ids"])) == 2
    assert payload["metadatas"][0] == {
        "document_id": "doc-1",
        "page": 2,
        "chunk_index": 4,
        "page_chunk_index": 1,
        "source_type": "text",
    }


def test_empty_index_does_not_call_embedding_service(monkeypatch):
    monkeypatch.setattr(embedding, "Chroma", FakeVectorStore)

    def unexpected(_texts):
        raise AssertionError("embedding should not run for no chunks")

    monkeypatch.setattr(type(embedding.embeddings), "embed_documents", unexpected)

    store = embedding.build_vectorstore([])

    assert not hasattr(store._collection, "payload")
