from types import SimpleNamespace

from multimodal_rag import answer_generator


def test_unknown_citation_is_preserved_and_prevents_verified_status(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        answer_generator,
        "retrieve_multimodal",
        lambda *args, **kwargs: {
            "text": [{
                "source_id": "T1",
                "document_id": "doc-1",
                "page": 1,
                "chunk_index": 0,
                "text": "The encoder has six layers.",
            }],
            "images": [],
        },
    )
    answer = "The encoder has six layers [T1]. The decoder has eight layers [T99]."
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=answer)
    )])
    monkeypatch.setattr(
        answer_generator.client.chat.completions,
        "create",
        lambda **kwargs: response,
    )
    monkeypatch.setattr(
        answer_generator,
        "verify_claims",
        lambda **kwargs: [{
            "claim": "The encoder has six layers",
            "citation_ids": ["T1"],
            "verdict": "supported",
            "explanation": "T1 states six layers.",
        }],
    )

    result = answer_generator.generate_answer("How many layers?")

    assert result["answer"] == answer
    assert result["invalid_citations"] == ["T99"]
    assert result["verification_status"] == "completed_with_invalid_citations"
    assert result["claim_support_status"] == "invalid_citations_present"
