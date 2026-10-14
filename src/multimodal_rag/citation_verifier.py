
import json
import logging
import os
import base64
import mimetypes
import re
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL_NAME = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemini-2.5-flash",
)


# --------------------------------------------------
# IMAGE ENCODING
# --------------------------------------------------

def encode_image(image_path):
    """Encode an image as a base64 data URL."""
    image_path = Path(image_path)

    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"

    with image_path.open("rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# --------------------------------------------------
# CITATION ASSOCIATION
# --------------------------------------------------

CITATION_PATTERN = re.compile(
    r"\[((?:[TI]\d+)(?:\s*,\s*[TI]\d+)*)\]"
)

CITATION_ID_PATTERN = re.compile(r"\b[TI]\d+\b")


def parse_citations(answer, available_source_ids):
    """Return recognized citation IDs and malformed citation-like tokens."""
    citation_tokens = re.findall(r"\[[^\]]*\]", answer or "")
    cited_ids = [
        source_id
        for token in citation_tokens
        for source_id in CITATION_ID_PATTERN.findall(token)
    ]
    available = set(available_source_ids)
    malformed = [
        token for token in re.findall(r"\[[^\]]*\]", answer or "")
        if re.search(r"[A-Za-z]\s*\d", token)
        and not CITATION_PATTERN.fullmatch(token)
    ]
    return {
        "cited_ids": cited_ids,
        "invalid_ids": sorted(set(cited_ids) - available),
        "valid_ids": list(dict.fromkeys(
            source_id for source_id in cited_ids if source_id in available
        )),
        "malformed_citations": malformed,
    }



def normalize_claim(text):
    """Normalize text for matching a verified claim to the answer."""
    text = CITATION_PATTERN.sub("", text)
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.lower().split()).strip()


def get_original_citations(answer):
    """
    Associate citation IDs with sentences in the original answer.

    Example:
        The encoder has six layers [T5].
        The decoder has six layers [T7].

    T5 is associated with the encoder sentence, while T7 is
    associated with the decoder sentence.
    """
    mapping = []

    segments = re.split(r"(?<=[.!?])\s+|\n+", answer)

    for segment in segments:
        ids = [
            source_id
            for group in CITATION_PATTERN.findall(segment)
            for source_id in CITATION_ID_PATTERN.findall(group)
        ]

        if segment.strip():
            mapping.append({
                "text": normalize_claim(segment),
                "original_text": segment.strip(),
                "citation_ids": set(ids),
            })

    return mapping


def citations_for_claim(claim, citation_mapping):
    """
    Return citation IDs associated with a verifier claim.

    A citation is allowed only if the normalized verifier claim
    matches text from a sentence carrying that citation.
    """
    normalized = normalize_claim(claim)

    if not normalized:
        return set()

    matches = [
        entry["citation_ids"]
        for entry in citation_mapping
        if normalized in entry["text"]
    ]

    if not matches:
        return set()

    # If multiple sentences match, allow only IDs common to all.
    return set.intersection(*matches)


# --------------------------------------------------
# CLAIM VERIFICATION
# --------------------------------------------------

def verify_claims(answer, sources):
    """
    Verify factual claims against retrieved text and image evidence.

    Checks:
    1. The source ID exists in the supplied evidence.
    2. The source ID was associated with the claim in the original answer.
    3. The model's verdict is one of the supported verdict values.

    Note: The model's verdict is an evidence assessment, not a
    mathematical guarantee that the claim is correct.
    """
    if not sources:
        return []

    message_content = []
    citation_mapping = get_original_citations(answer)
    sources_by_id = {item["source_id"]: item for item in sources}
    valid_source_ids = set(sources_by_id)
    claim_evidence = []

    # Each answer sentence receives only sources cited on that sentence.
    # This prevents unrelated retrieved text from being presented as support.
    for sentence in citation_mapping:
        sentence_sources = []
        for source_id in sorted(sentence["citation_ids"] & valid_source_ids):
            source = sources_by_id[source_id]
            item = {
                "source_id": source_id,
                "type": source["type"],
                "page": source["page"],
            }
            if source["type"] == "text":
                item["content"] = source["text"]
            elif source["type"] == "image":
                item["content"] = "The associated image is attached."
                message_content.append({
                    "type": "text",
                    "text": (
                        f"Image evidence for answer sentence "
                        f"{sentence['original_text']!r}, source [{source_id}]:"
                    ),
                })
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": encode_image(source["image_path"])},
                })
            sentence_sources.append(item)
        claim_evidence.append({
            "answer_sentence": sentence["original_text"],
            "cited_source_ids": sorted(sentence["citation_ids"]),
            "evidence": sentence_sources,
        })

    prompt = f"""
You are a careful citation verifier.

Check whether each factual claim in the answer is supported
by the supplied evidence.

Answer sentences and their associated evidence:
{json.dumps(claim_evidence, ensure_ascii=False)}

Return only a valid JSON array. Each item must contain:
- claim: a factual claim from the answer
- citation_ids: a list of source IDs cited for that claim
- verdict: "supported", "contradicted", or "insufficient_evidence"
- explanation: a brief reason

Rules:
- Evaluate claims using only the evidence listed for that answer sentence.
- Do not use evidence from another sentence's entry.
- Use only source IDs supplied in that sentence's evidence entry.
- If a sentence has no associated evidence, any factual claim in it must be
  "insufficient_evidence" with an explanation that no cited evidence was supplied.
- Use only citation IDs actually associated with each claim
  in the original answer.
- A source ID existing does not prove that it supports a claim.
- Use "supported" only when the supplied evidence supports the claim.
- Use "contradicted" only when the evidence conflicts with the claim.
- Use "insufficient_evidence" when the evidence does not settle it.
- For visual claims, inspect the attached image associated with its source ID.
- Image paths and metadata alone are not evidence of visual content.
- Do not use general world knowledge to fill evidence gaps.
- Do not invent sources or details.
- Return JSON only, without Markdown fences.
"""

    message_content.insert(0, {
        "type": "text",
        "text": prompt,
    })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ],
        temperature=0,
        max_tokens=600,
    )

    raw = response.choices[0].message.content or ""
    raw = raw.strip()

    # Remove Markdown fences if returned by the model.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]

        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].strip()

    # Parse JSON.
    try:
        results = json.loads(raw)

    except json.JSONDecodeError as exc:
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        logger.error(
            "Verifier returned invalid JSON (finish_reason=%s): %.500s",
            finish_reason,
            raw,
        )

        raise ValueError(
            "The citation verifier returned invalid or incomplete JSON "
            f"(finish_reason={finish_reason!r})."
        ) from exc

    if not isinstance(results, list):
        raise ValueError(
            "The citation verifier response must be a JSON array."
        )

    valid_verdicts = {
        "supported",
        "contradicted",
        "insufficient_evidence",
    }

    # Validate each verification result.
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(
                "Each verification result must be a JSON object."
            )

        required_fields = {
            "claim",
            "citation_ids",
            "verdict",
            "explanation",
        }

        if not required_fields.issubset(result):
            raise ValueError(
                "A verification result is missing required fields."
            )

        if not isinstance(result["claim"], str):
            raise ValueError("The claim field must be a string.")

        if not isinstance(result["explanation"], str):
            raise ValueError("The explanation field must be a string.")

        if result["verdict"] not in valid_verdicts:
            raise ValueError(
                f"Invalid verification verdict: {result['verdict']}"
            )

        if not isinstance(result["citation_ids"], list):
            raise ValueError("citation_ids must be a list.")

        original_ids = result["citation_ids"]

        # First: reject source IDs that do not exist in the evidence.
        valid_ids = {
            source_id
            for source_id in original_ids
            if isinstance(source_id, str)
            and source_id in valid_source_ids
        }

        unknown_ids = {
            source_id
            for source_id in original_ids
            if not isinstance(source_id, str)
            or source_id not in valid_source_ids
        }

        if unknown_ids:
            print(
                "Warning: unknown citation IDs",
                unknown_ids,
                "for claim:",
                result["claim"],
            )

        # Second: reject valid sources not attached to this claim
        # in the original answer.
        allowed_ids = citations_for_claim(
            result["claim"],
            citation_mapping,
        )

        unexpected_ids = valid_ids - allowed_ids
        valid_ids = valid_ids & allowed_ids

        result["citation_ids"] = sorted(valid_ids)

        if unexpected_ids:
            print(
                "Warning: citations not attached to this claim:",
                unexpected_ids,
                "for claim:",
                result["claim"],
            )

        # No validated citation means the claim cannot remain verified.
        if not valid_ids:
            result["verdict"] = "insufficient_evidence"
            result["explanation"] = (
                "No citation associated with this claim in the "
                "original answer could be validated."
            )

    return results
