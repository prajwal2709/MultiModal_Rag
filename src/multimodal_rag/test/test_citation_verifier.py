
import json
from types import SimpleNamespace

from multimodal_rag import citation_verifier


def test_contradicted_claim_keeps_its_verdict(monkeypatch):
    # The answer makes a false claim but cites a real source.
    answer = (
        "The encoder consists of eight identical layers [T5]."
    )

    # The retrieved evidence says the encoder has six layers.
    sources = [
        {
            "source_id": "T5",
            "type": "text",
            "page": 2,
            "text": (
                "The encoder is composed of a stack "
                "of N = 6 identical layers."
            ),
        }
    ]

    # Mock the model response so this test makes no API request.
    mock_result = [
        {
            "claim": (
                "The encoder consists of eight identical layers"
            ),
            "citation_ids": ["T5"],
            "verdict": "contradicted",
            "explanation": (
                "The evidence states that the encoder has six layers."
            ),
        }
    ]

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(mock_result)
                )
            )
        ]
    )

    def fake_create(**kwargs):
        return mock_response

    monkeypatch.setattr(
        citation_verifier.client.chat.completions,
        "create",
        fake_create,
    )

    results = citation_verifier.verify_claims(answer, sources)

    assert len(results) == 1
    assert results[0]["verdict"] == "contradicted"
    assert results[0]["citation_ids"] == ["T5"]

def test_supported_claim_keeps_its_verdict(monkeypatch):
    answer = "The encoder consists of six identical layers [T5]."

    sources = [
        {
            "source_id": "T5",
            "type": "text",
            "page": 2,
            "text": "The encoder consists of N = 6 identical layers.",
        }
    ]

    mock_result = [
        {
            "claim": "The encoder consists of six identical layers",
            "citation_ids": ["T5"],
            "verdict": "supported",
            "explanation": "The evidence confirms six layers.",
        }
    ]

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(mock_result)
                )
            )
        ]
    )

    monkeypatch.setattr(
        citation_verifier.client.chat.completions,
        "create",
        lambda **kwargs: mock_response,
    )

    results = citation_verifier.verify_claims(answer, sources)

    assert results[0]["verdict"] == "supported"
    assert results[0]["citation_ids"] == ["T5"]


def test_insufficient_evidence_keeps_its_verdict(monkeypatch):
    answer = "The encoder uses eight attention heads [T5]."

    sources = [
        {
            "source_id": "T5",
            "type": "text",
            "page": 2,
            "text": "The encoder consists of six identical layers.",
        }
    ]

    mock_result = [
        {
            "claim": "The encoder uses eight attention heads",
            "citation_ids": ["T5"],
            "verdict": "insufficient_evidence",
            "explanation": "The evidence does not specify the number of attention heads.",
        }
    ]

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(mock_result)
                )
            )
        ]
    )

    monkeypatch.setattr(
        citation_verifier.client.chat.completions,
        "create",
        lambda **kwargs: mock_response,
    )

    results = citation_verifier.verify_claims(answer, sources)

    assert results[0]["verdict"] == "insufficient_evidence"
    assert results[0]["citation_ids"] == ["T5"]


def test_uncited_factual_sentence_cannot_inherit_neighboring_citation(monkeypatch):
    answer = (
        "The encoder consists of six identical layers [T5]. "
        "The GPU serial number is SN-123."
    )
    sources = [{
        "source_id": "T5",
        "type": "text",
        "page": 2,
        "text": "The encoder consists of N = 6 identical layers.",
    }]
    mock_result = [
        {
            "claim": "The encoder consists of six identical layers",
            "citation_ids": ["T5"],
            "verdict": "supported",
            "explanation": "T5 states six layers.",
        },
        {
            "claim": "The GPU serial number is SN-123",
            "citation_ids": ["T5"],
            "verdict": "supported",
            "explanation": "The verifier incorrectly reused T5.",
        },
    ]
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(mock_result))
    )])
    monkeypatch.setattr(
        citation_verifier.client.chat.completions,
        "create",
        lambda **kwargs: response,
    )

    results = citation_verifier.verify_claims(answer, sources)

    assert len(results) == 2
    assert results[1]["citation_ids"] == []
    assert results[1]["verdict"] == "insufficient_evidence"


def test_malformed_verifier_json_raises_clear_error(monkeypatch):
    answer = "The encoder has six layers [T5]."
    sources = [{
        "source_id": "T5",
        "type": "text",
        "page": 2,
        "text": "The encoder has N = 6 layers.",
    }]
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="not json", finish_reason="stop")
    )])
    monkeypatch.setattr(
        citation_verifier.client.chat.completions,
        "create",
        lambda **kwargs: response,
    )

    import pytest
    with pytest.raises(ValueError, match="invalid or incomplete JSON"):
        citation_verifier.verify_claims(answer, sources)
