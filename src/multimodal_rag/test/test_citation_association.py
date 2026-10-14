import json
from types import SimpleNamespace

from multimodal_rag import citation_verifier
from multimodal_rag.citation_verifier import (
    get_original_citations,
    citations_for_claim,
    parse_citations,
)


def test_correct_citation_is_associated():
    answer = (
        "The encoder consists of six identical layers [T5]. "
        "The decoder also consists of six layers [T7]."
    )

    mapping = get_original_citations(answer)

    assert citations_for_claim(
        "The encoder consists of six identical layers.",
        mapping,
    ) == {"T5"}


def test_decoder_citation_is_not_encoder_citation():
    answer = (
        "The encoder consists of six identical layers [T5]. "
        "The decoder also consists of six layers [T7]."
    )

    mapping = get_original_citations(answer)

    assert "T7" not in citations_for_claim(
        "The encoder consists of six identical layers.",
        mapping,
    )


def test_uncited_claim_has_no_associated_source():
    answer = "The encoder consists of six identical layers [T5]."

    mapping = get_original_citations(answer)

    assert citations_for_claim(
        "The encoder uses eight layers.",
        mapping,
    ) == set()


def test_uncited_answer_sentence_is_retained_for_verification():
    mapping = get_original_citations(
        "The encoder has six layers [T5]. The serial number is unavailable."
    )

    assert len(mapping) == 2
    assert mapping[1]["citation_ids"] == set()


def test_grouped_citations_are_preserved():
    answer = (
        "The architecture uses attention mechanisms [T3, T4]."
    )

    mapping = get_original_citations(answer)

    assert citations_for_claim(
        "The architecture uses attention mechanisms.",
        mapping,
    ) == {"T3", "T4"}


def test_two_claims_in_same_paragraph_get_separate_citations():
    answer = (
        "The encoder processes the input [T5]. "
        "The decoder generates the output [T7]."
    )

    mapping = get_original_citations(answer)

    encoder_ids = citations_for_claim(
        "The encoder processes the input.",
        mapping,
    )
    decoder_ids = citations_for_claim(
        "The decoder generates the output.",
        mapping,
    )

    assert encoder_ids == {"T5"}
    assert decoder_ids == {"T7"}


def test_paraphrased_claim_is_not_matched_automatically():
    answer = "The encoder consists of six identical layers [T5]."

    mapping = get_original_citations(answer)

    # Current matching is text-based, not semantic.
    # A paraphrase should not receive an unverified association.
    assert citations_for_claim(
        "The encoder contains six repeated blocks.",
        mapping,
    ) == set()

def test_wrong_layer_count_has_no_valid_association():
    answer = "The encoder consists of six identical layers [T5]."
    mapping = get_original_citations(answer)

    # The incorrect claim must not inherit T5 just because it
    # discusses the same subject.
    assert citations_for_claim(
        "The encoder consists of eight identical layers.",
        mapping,
    ) == set()


def test_wrong_citation_is_not_associated_with_claim():
    answer = (
        "The encoder consists of six identical layers [T5]. "
        "The decoder also consists of six layers [T7]."
    )
    mapping = get_original_citations(answer)

    assert "T7" not in citations_for_claim(
        "The encoder consists of six identical layers.",
        mapping,
    )


def test_image_citation_is_preserved():
    answer = "The encoder is shown on the left side of Figure 1 [I1]."
    mapping = get_original_citations(answer)

    assert citations_for_claim(
        "The encoder is shown on the left side of Figure 1.",
        mapping,
    ) == {"I1"}

def test_valid_citation_does_not_guarantee_supported_claim(monkeypatch):
    answer = "The encoder uses eight attention heads [T5]."

    # T5 exists, but its content discusses the number of layers,
    # not the number of attention heads.
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
            "claim": "The encoder uses eight attention heads",
            "citation_ids": ["T5"],
            "verdict": "insufficient_evidence",
            "explanation": (
                "T5 describes the number of layers, "
                "not the number of attention heads."
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

    monkeypatch.setattr(
        citation_verifier.client.chat.completions,
        "create",
        lambda **kwargs: mock_response,
    )

    results = citation_verifier.verify_claims(answer, sources)

    assert results[0]["citation_ids"] == ["T5"]
    assert results[0]["verdict"] == "insufficient_evidence"


def test_parser_reports_unknown_ids_without_rewriting_answer():
    answer = "The encoder has six layers [T2, T99]."
    report = parse_citations(answer, {"T2"})

    assert report["cited_ids"] == ["T2", "T99"]
    assert report["valid_ids"] == ["T2"]
    assert report["invalid_ids"] == ["T99"]
    assert answer.endswith("[T2, T99].")


def test_parser_detects_malformed_and_non_text_image_tokens():
    report = parse_citations(
        "See [T 2] and [X3]. Valid: [I1, T2].",
        {"I1", "T2"},
    )

    assert report["valid_ids"] == ["I1", "T2"]
    assert report["malformed_citations"] == ["[T 2]", "[X3]"]


def test_malformed_group_still_recognizes_valid_id_inside():
    report = parse_citations("Claim [T2, X3].", {"T2"})

    assert report["valid_ids"] == ["T2"]
    assert report["malformed_citations"] == ["[T2, X3]"]
