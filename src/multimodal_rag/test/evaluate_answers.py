
import json
from pathlib import Path

from multimodal_rag.answer_generator import generate_answer
from multimodal_rag.citation_verifier import (
    citations_for_claim,
    get_original_citations,
    parse_citations,
)
from multimodal_rag.answer_generator import MODEL_NAME


TEST_DIR = Path(__file__).parent
DATASET_PATH = TEST_DIR / "answer_evaluation_dataset.json"
TEST_CASES = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
RESULTS_PATH = TEST_DIR / "answer_evaluation_results.json"


def score_answer(
    case, answer, verification, sources=None, verification_failed=False
):
    text = answer.lower()

    missing_evidence_phrases = [
        "cannot be determined",
        "not provided",
        "insufficient evidence",
        "unknown",
        "cannot determine",
        "not specified",
        "cannot be confirmed",
        "cannot confirm",
        "not established",
    ]

    acknowledges_serial_missing = (
        "serial number" in text
        and any(phrase in text for phrase in missing_evidence_phrases)
    )

    acknowledges_price_missing = (
        "price" in text
        and any(phrase in text for phrase in missing_evidence_phrases)
    )

    case_id = case["id"]
    extraneous_detail_detected = False
    matched_groups = None
    required_sources_present = None

    if case_id == "A001":
        answer_correct = (
            "6" in text
            and "layer" in text
            and "encoder" in text
        )
        answer_relevant = answer_correct
        evidence_acknowledged = True

    elif case_id == "A002":
        answer_correct = (
            "p100" in text
            and acknowledges_serial_missing
        )
        answer_relevant = answer_correct
        evidence_acknowledged = acknowledges_serial_missing

    elif case_id == "A003":
        answer_correct = acknowledges_serial_missing
        extraneous_detail_detected = "p100" in text
        answer_relevant = (
            answer_correct and not extraneous_detail_detected
        )
        evidence_acknowledged = acknowledges_serial_missing

    elif case_id == "A004":
        answer_correct = (
            acknowledges_serial_missing
            and acknowledges_price_missing
        )
        answer_relevant = answer_correct
        evidence_acknowledged = answer_correct


    elif case_id == "A005":
        acknowledges_serial_missing = (
            "serial number" in text
            and any(phrase in text for phrase in missing_evidence_phrases)
        )
        answer_correct = (
            "p100" in text
            and ("8" in text or "eight" in text)
            and acknowledges_serial_missing
        )
        answer_relevant = answer_correct
        evidence_acknowledged = acknowledges_serial_missing

    elif case.get("expected_fact_groups"):
        matched_groups = [
            any(str(expected).lower() in text for expected in group)
            for group in case["expected_fact_groups"]
        ]
        source_types = {item.get("type") for item in (sources or [])}
        required_source_types = set(case.get("required_source_types", []))
        required_sources_present = required_source_types.issubset(source_types)
        answer_correct = all(matched_groups) and required_sources_present
        answer_relevant = answer_correct
        evidence_acknowledged = answer_correct


    else:
        answer_correct = False
        answer_relevant = False
        evidence_acknowledged = False

    supported_claim_count = None if verification_failed else sum(
        item.get("verdict") == "supported" for item in verification
    )
    insufficient_evidence_claim_count = None if verification_failed else sum(
        item.get("verdict") == "insufficient_evidence" for item in verification
    )
    contradicted_claim_count = None if verification_failed else sum(
        item.get("verdict") == "contradicted" for item in verification
    )

    return {
        "answer_correct": answer_correct,
        "evidence_acknowledged": evidence_acknowledged,
        "answer_relevant": answer_relevant,
        "extraneous_detail_detected": extraneous_detail_detected,
        "expected_fact_groups": case.get("expected_fact_groups"),
        "expected_fact_groups_matched": matched_groups,
        "required_source_types": case.get("required_source_types"),
        "required_source_types_present": required_sources_present,
        "supported_claim_count": supported_claim_count,
        "insufficient_evidence_claim_count": insufficient_evidence_claim_count,
        "contradicted_claim_count": contradicted_claim_count,
    }


def evaluate_case(case):
    result = generate_answer(
        case["question"],
        text_k=3,
        image_k=2,
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    verification = result.get("verification", [])
    verification_failed = result.get("verification_status") == "verification_failed"
    scores = score_answer(
        case,
        answer,
        verification,
        sources=sources,
        verification_failed=verification_failed,
    )

    valid_source_ids = {
        source["source_id"]
        for source in sources
    }

    # Check that every citation in the answer refers to a
    # source returned by the application.
    citation_report = parse_citations(answer, valid_source_ids)
    invalid_citations = sorted(
        set(citation_report["invalid_ids"])
        | set(result.get("invalid_citations", []))
    )
    citation_mapping = get_original_citations(answer)
    associated_ids = set().union(
        *(item["citation_ids"] for item in citation_mapping)
    ) if citation_mapping else set()
    checked_claim_associations = [
        set(item.get("citation_ids", [])) <= citations_for_claim(
            item.get("claim", ""), citation_mapping
        )
        and bool(item.get("citation_ids"))
        for item in verification
    ]
    citation_claim_association = (
        None if not verification else
        "complete" if all(checked_claim_associations) else
        "partial" if any(checked_claim_associations) else
        "unassociated"
    )

    verdict_counts = {
        "supported": 0,
        "contradicted": 0,
        "insufficient_evidence": 0,
    }

    for item in verification:
        verdict = item.get("verdict")

        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    if verification_failed:
        claim_support_status = "verification_failed"
    elif not verification:
        claim_support_status = "no_claims_verified"
    elif verdict_counts["contradicted"] > 0:
        claim_support_status = "contains_contradicted_claims"
    elif verdict_counts["insufficient_evidence"] > 0:
        claim_support_status = "contains_unverified_claims"
    else:
        claim_support_status = "all_evaluated_claims_supported"


    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answer": answer,
        "scores": scores,
        "claim_support_status": claim_support_status,
        "answer_claim_support_status": result.get(
            "claim_support_status", "missing"
        ),
        "claim_verdict_counts": None if verification_failed else verdict_counts,
        "expected_fact": case["expected_fact"],
        "verification_status": result.get(
            "verification_status", "missing"
        ),
        "verification_error": result.get("verification_error"),
        "verification_error_code": result.get("verification_error_code"),
        "verification_summary": (
            None if verification_failed else result.get("verification_summary", {})
        ),
        "citation_validity": (
            None if not citation_report["cited_ids"]
            and not citation_report["malformed_citations"] else
            "valid" if not invalid_citations
            and not citation_report["malformed_citations"] else "invalid"
        ),
        "invalid_citations": invalid_citations,
        "malformed_citations": citation_report["malformed_citations"],
        "citation_claim_association": citation_claim_association,
        "claims_with_associated_citations": sum(checked_claim_associations),
        "claims_without_associated_citations": sum(
            not associated for associated in checked_claim_associations
        ),
        "verification": verification,
        "retrieved_source_count": result.get(
            "retrieved_source_count", 0
        ),
    }

def main():
    results = []
    errors = []

    print("\n========== FULL RAG ANSWER EVALUATION ==========\n")

    for case in TEST_CASES:
        print(f"Running {case['id']}: {case['category']}")

        try:
            result = evaluate_case(case)
            results.append(result)

            print("Question:", result["question"])
            print("Answer:", result["answer"])
            print("Verification status:", result["verification_status"])
            print("Citation validity:", result["citation_validity"])
            print("Claim support status:", result["claim_support_status"])
            print("Claim verdict counts:", result["claim_verdict_counts"])

            print("Claim-level verification:")

            for check in result["verification"]:
                print("  Claim:", check.get("claim", "Unknown claim"))
                print("  Sources:", check.get("citation_ids", []))
                print("  Verdict:", check.get("verdict", "unknown"))
                print("  Explanation:", check.get("explanation", ""))

            scores = result["scores"]

            print(f"Answer behavior passed: {scores['answer_correct']}")
            print(
                "Evidence acknowledgment passed: "
                f"{scores['evidence_acknowledged']}"
            )
            print(f"Answer relevance passed: {scores['answer_relevant']}")
            print(
                "Extraneous detail detected: "
                f"{scores['extraneous_detail_detected']}"
            )
            print("-" * 60)


        except Exception as exc:
            errors.append({
                "id": case["id"],
                "error": str(exc),
            })
            print("ERROR:", exc)
            print("-" * 60)

    scored_cases = [
        result for result in results
        if "scores" in result
    ]

    answer_correct_count = sum(
        result["scores"]["answer_correct"]
        for result in scored_cases
    )

    abstention_correct_count = sum(
        result["scores"]["evidence_acknowledged"]
        for result in scored_cases
    )

    relevance_correct_count = sum(
        result["scores"]["answer_relevant"]
        for result in scored_cases
    )

    extraneous_detail_count = sum(
        result["scores"]["extraneous_detail_detected"]
        for result in scored_cases
    )

    claim_evaluated_cases = [
        result for result in results
        if isinstance(result.get("claim_verdict_counts"), dict)
    ]
    verdict_totals = {
        verdict: sum(
            result["claim_verdict_counts"].get(verdict, 0)
            for result in claim_evaluated_cases
        )
        for verdict in ("supported", "contradicted", "insufficient_evidence")
    } if claim_evaluated_cases else None
    citation_results = [
        result["citation_validity"] for result in results
        if result["citation_validity"] is not None
    ]
    association_results = [
        result["citation_claim_association"] for result in results
        if result["citation_claim_association"] is not None
    ]
    aggregate_metrics = {
        "answer_correct": {
            "passed": answer_correct_count,
            "total": len(scored_cases),
        },
        "answer_relevance": {
            "passed": relevance_correct_count,
            "total": len(scored_cases),
        },
        "evidence_acknowledgment": {
            "passed": abstention_correct_count,
            "total": len(scored_cases),
        },
        "cases_with_extraneous_details": extraneous_detail_count,
        "citation_id_validity": {
            "valid_cases": sum(value == "valid" for value in citation_results),
            "invalid_cases": sum(value == "invalid" for value in citation_results),
            "not_applicable_cases": len(results) - len(citation_results),
        },
        "citation_claim_association": {
            "complete_cases": sum(value == "complete" for value in association_results),
            "partial_cases": sum(value == "partial" for value in association_results),
            "unassociated_cases": sum(value == "unassociated" for value in association_results),
            "not_applicable_cases": len(results) - len(association_results),
        },
        "claim_verdict_counts": verdict_totals,
        "unsupported_claims": (
            verdict_totals["insufficient_evidence"] if verdict_totals else None
        ),
        "contradicted_claims": (
            verdict_totals["contradicted"] if verdict_totals else None
        ),
        "retrieval_failures": sum(
            result.get("retrieved_source_count", 0) == 0 for result in results
        ),
        "evaluation_errors": len(errors),
        "verification_failures": sum(
            result.get("verification_status") == "verification_failed"
            for result in results
        ),
        "claim_verdict_cases_available": len(claim_evaluated_cases),
    }


    print("\n========== SCORING SUMMARY ==========")
    print(
        f"Answer behavior passed: "
        f"{answer_correct_count}/{len(scored_cases)}"
    )
    print(
        f"Evidence acknowledgment passed: "
        f"{abstention_correct_count}/{len(scored_cases)}"
    )

    print(
        f"Answer relevance passed: "
        f"{relevance_correct_count}/{len(scored_cases)}"
    )
    print(
        f"Cases with extraneous details: "
        f"{extraneous_detail_count}/{len(scored_cases)}"
    )


    report = {
        "total_cases": len(TEST_CASES),
        "evaluated_cases": len(results),
        "scored_cases": len(scored_cases),
        "answer_behavior_passed": answer_correct_count,
        "evidence_acknowledgment_passed": abstention_correct_count,
        "errors": errors,
        "results": results,
        "answer_relevance_passed": relevance_correct_count,
        "cases_with_extraneous_details": extraneous_detail_count,
        "aggregate_metrics": aggregate_metrics,
        "configuration": {
            "model": MODEL_NAME,
            "dataset": str(DATASET_PATH.name),
            "dataset_size": len(TEST_CASES),
            "text_k": 3,
            "image_k": 2,
            "scorer": "deterministic case-specific rules; not an independent truth guarantee",
        },
    }

    RESULTS_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== SUMMARY ==========")
    print("Test cases:", len(TEST_CASES))
    print("Evaluated:", len(results))
    print("Errors:", len(errors))
    print("Saved results:", RESULTS_PATH)


if __name__ == "__main__":
    main()
