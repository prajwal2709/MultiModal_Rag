
import os
import base64
import mimetypes
from pathlib import Path
import logging
from multimodal_rag.citation_verifier import verify_claims, parse_citations
from dotenv import load_dotenv
from openai import OpenAI

from multimodal_rag.retrival import retrieve_multimodal

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


def encode_image(image_path):
    image_path = Path(image_path)

    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def generate_answer(question, text_k=3, image_k=2, progress_callback=None, retrieval_hint=None):
    def report_progress(phase):
        if progress_callback:
            progress_callback(phase)

    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError(
            "OPENROUTER_API_KEY is missing from your .env file."
        )

    report_progress("retrieving")
    retrieval_query = question
    if retrieval_hint:
        retrieval_query = f"{question}\nSelected evidence retrieval hints: {retrieval_hint}"
    retrieved = retrieve_multimodal(
        retrieval_query,
        text_k=text_k,
        image_k=image_k,
    )
    report_progress("generating")

    # Map every valid source ID to its real source information.
    source_map = {}

    text_sections = []

    for item in retrieved["text"]:
        source_id = item["source_id"]

        source_map[source_id] = {
            "type": "text",
            "source_type": "text",
            "document_id": item.get("document_id", "unknown"),
            "page": item["page"],
            "chunk_index": item.get("chunk_index"),
            "text": item["text"],
        }

        text_sections.append(
            f"[{source_id}] PDF page {item['page']}\n"
            f"{item['text']}"
        )

    image_sections = []

    for item in retrieved["images"]:
        source_id = item["source_id"]

        source_map[source_id] = {
            "type": "image",
            "source_type": "image",
            "document_id": item.get("document_id", "unknown"),
            "image_id": item.get("image_id"),
            "page": item["page"],
            "image_path": item["image_path"],
            "distance": item["distance"],
            "rank_score": item.get("rank_score", item["distance"]),
            "text_page_distance": item.get("text_page_distance"),
        }

        image_sections.append(
            f"{source_id}: PDF page {item['page']}"
        )

    text_context = "\n\n".join(text_sections)
    if not text_context:
        text_context = "No relevant text was retrieved."

    image_context = "\n".join(image_sections)
    if not image_context:
        image_context = "No images were retrieved."


    prompt = f"""
You are a careful research assistant answering questions about a PDF.

Your task is to answer the user's question using ONLY the
retrieved text and images supplied below.

Evidence rules:
1. Every factual claim must be supported by the supplied evidence.
2. Cite each factual claim using the exact source IDs provided.
3. Use [Tn] for claims supported by retrieved text.
4. Use [In] for claims supported by visual information.
5. Prefer text citations for exact numerical specifications.
6. Use image citations for visible layouts, labels, arrows,
   and spatial relationships. A claim about where an element appears
   in a diagram must cite an image source, not a text passage.
7. Never invent source IDs, page numbers, or factual details.
8. A valid citation does not automatically mean its source
   supports the associated claim.
9. If the evidence does not answer the question, explicitly
   say that the answer cannot be determined from the retrieved
   evidence. Do not guess or use outside knowledge.
10. If only part of the question is answerable, answer that
    part with citations and clearly identify what remains unknown.
11. Do not claim that information is absent from the entire PDF
    merely because it is absent from the retrieved excerpts.
12. Keep the answer concise, clear, and properly spaced.
13. Correct a false premise only when the supplied evidence supports the correction.
14. Do not add unrelated details merely because they appear in a retrieved chunk.

Retrieved text:
{text_context}

Retrieved images:
{image_context}

Question:
{question}

Write the answer with inline citations.
Do not create a separate source list; the application will build it.
"""

    message_content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for item in retrieved["images"]:
        message_content.append({
            "type": "text",
            "text": (
                f"Image source [{item['source_id']}], "
                f"PDF page {item['page']}:"
            ),
        })

        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": encode_image(item["image_path"]),
            },
        })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ],
        temperature=0.2,
        # Keep the completion request under the smaller OpenRouter key budgets
        # used by this project; concise answers are preferred in the UI.
        max_tokens=900,
    )

    answer = response.choices[0].message.content or ""
    report_progress("verifying")

    # Extract citation IDs from both single and grouped citations.
    # Supports [T2], [T3, T4], [T2, I1], etc.
    # Validate citation IDs against retrieved sources.
    valid_ids = set(source_map)
    citation_report = parse_citations(answer, valid_ids)
    cited_ids = citation_report["cited_ids"]
    invalid_ids = set(citation_report["invalid_ids"])
    if invalid_ids:
        logger.warning("Answer contains unknown citation IDs: %s", sorted(invalid_ids))

    # Preserve malformed citation-like tokens for review. A bad citation
    # must remain visible instead of being silently stripped from the answer.
    malformed_citations = citation_report["malformed_citations"]


    # Build the source list from valid citations in the answer.
    verified_ids = citation_report["valid_ids"]

    sources = [
        {
            "source_id": source_id,
            **source_map[source_id],
        }
        for source_id in verified_ids
    ]

    # Verify whether the cited evidence supports each claim.
    if sources:
        try:
            verification = verify_claims(answer=answer, sources=sources)
            verification_error = None
            verification_error_code = None
        except Exception as error:
            logger.error(
                "Claim verification failed (%s, code=%s)",
                type(error).__name__,
                getattr(error, "status_code", None),
            )
            verification = []
            verification_error = type(error).__name__
            verification_error_code = getattr(error, "status_code", None)

        if verification_error is not None:
            verification_status = "verification_failed"
        elif not verification:
            verification_status = "no_claims_verified"
        elif any(
            item.get("verdict") == "insufficient_evidence"
            for item in verification
        ):
            verification_status = "completed_with_insufficient_evidence"
        elif any(
            item.get("verdict") == "contradicted"
            for item in verification
        ):
            verification_status = "completed_with_contradictions"
        else:
            verification_status = "completed"

        if not verification_error and (invalid_ids or malformed_citations):
            verification_status = "completed_with_invalid_citations"

    else:
        verification = []
        verification_error = None
        verification_error_code = None
        verification_status = (
            "completed_with_invalid_citations"
            if invalid_ids or malformed_citations
            else "skipped_no_cited_sources"
        )


    verdict_counts = {
        "supported": 0,
        "contradicted": 0,
        "insufficient_evidence": 0,
    }

    for result in verification:
        verdict = result.get("verdict")
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    retrieved_sources = [
        {"source_id": source_id, **source_info}
        for source_id, source_info in source_map.items()
    ]

    result = {
    "answer": answer,
    "sources": sources,
    "retrieved_sources": retrieved_sources,
    "retrieved_source_count": len(source_map),
    "invalid_citations": sorted(invalid_ids),
    "malformed_citations": malformed_citations,
    "verification": verification,
    "verification_error": verification_error,
    "verification_error_code": verification_error_code,
    "verification_summary": verdict_counts,
    "verification_status": verification_status,
    "claim_support_status": (
        "verification_failed" if verification_error else
        "invalid_citations_present" if invalid_ids or malformed_citations else
        "no_claims_verified" if not verification else
        "contains_contradicted_claims" if verdict_counts["contradicted"] else
        "contains_unverified_claims" if verdict_counts["insufficient_evidence"] else
        "all_evaluated_claims_supported"
    ),
}
    report_progress("complete")
    return result


if __name__ == "__main__":

    question = (
    "What was the exact serial number of the GPU used "
    "to train the original Transformer model?"
)

    result = generate_answer(question)

    print("\n========== FINAL ANSWER ==========\n")
    print(result["answer"])

    print("\n========== VERIFIED SOURCES ==========\n")

    # Print sources used by the generated answer.
    for source in result["sources"]:
        print(
            f"[{source['source_id']}] "
            f"Type: {source['type']} | "
            f"PDF page: {source['page']}"
        )

        if source["type"] == "image":
            print("Image:", source["image_path"])
        else:
            print("Text:", source["text"][:250])

        print("-" * 50)

    print("\nRetrieved sources:", result["retrieved_source_count"])
    print("Invalid citations:", result["invalid_citations"])


    print("\n========== VERIFICATION SUMMARY ==========\n")

    for verdict, count in result["verification_summary"].items():
        print(f"{verdict}: {count}")

    print("\n========== VERIFICATION STATUS ==========")
    print(result["verification_status"])

    print("\n========== CLAIM-LEVEL VERIFICATION ==========\n")

    for check in result["verification"]:
        print("Claim:", check.get("claim", "Unknown claim"))
        print("Sources:", check.get("citation_ids", []))
        print("Verdict:", check.get("verdict", "unknown"))
        print("Explanation:", check.get("explanation", ""))
        print("-" * 50)
