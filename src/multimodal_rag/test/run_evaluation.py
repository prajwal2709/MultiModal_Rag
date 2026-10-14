
import json
from pathlib import Path

from multimodal_rag import citation_verifier


TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parents[2]

DATASET_PATH = TEST_DIR / "evaluation_dataset.json"
RESULTS_PATH = TEST_DIR / "evaluation_results.json"

VALID_VERDICTS = {
    "supported",
    "contradicted",
    "insufficient_evidence",
}


def parse_json_response(raw):
    """Parse and validate the model's JSON response."""
    raw = (raw or "").strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].strip()

    result = json.loads(raw)

    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object.")

    if result.get("verdict") not in VALID_VERDICTS:
        raise ValueError(
            f"Invalid verdict returned: {result.get('verdict')}"
        )

    if not isinstance(result.get("explanation"), str):
        raise ValueError("Model response must contain an explanation.")

    return result


def evaluate_case(case):
    """Evaluate one claim using text and/or actual image evidence."""
    evidence_descriptions = []
    image_parts = []

    for item in case["evidence"]:
        source_id = item["source_id"]
        source_type = item["type"]

        if source_type == "text":
            evidence_descriptions.append(
                f"Source ID: {source_id}\n"
                f"Source type: text\n"
                f"Evidence: {item['content']}"
            )

        elif source_type == "image":
            image_path = PROJECT_ROOT / item["image_path"]

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image evidence not found: {image_path}"
                )

            evidence_descriptions.append(
                f"Source ID: {source_id}\n"
                "Source type: image\n"
                "The actual image is attached separately. "
                "Inspect that image to assess the claim."
            )

            image_parts.append({
                "type": "text",
                "text": f"Image evidence [{source_id}]:",
            })

            image_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": citation_verifier.encode_image(image_path),
                },
            })

        else:
            raise ValueError(
                f"Unsupported evidence type: {source_type}"
            )

    evidence_text = "\n\n".join(evidence_descriptions)

    prompt = f"""
You are evaluating a factual claim against supplied document evidence.

Evidence:
{evidence_text}

Claim:
{case['claim']}

Choose exactly one verdict:
- supported: the evidence supports the claim.
- contradicted: the evidence directly conflicts with the claim.
- insufficient_evidence: the evidence does not establish whether
  the claim is true.

Rules:
- Use only the supplied evidence.
- Do not use outside knowledge.
- A source existing does not mean it supports the claim.
- For image evidence, inspect the actual attached image.
- Do not infer image contents from its filename or metadata.
- Do not claim to have inspected an image unless image data was supplied.
- Return only valid JSON with these fields:
  "verdict": one of the three allowed verdicts
  "explanation": a brief evidence-based explanation
"""

    message_content = [
        {
            "type": "text",
            "text": prompt,
        },
        *image_parts,
    ]

    response = citation_verifier.client.chat.completions.create(
        model=citation_verifier.MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ],
        temperature=0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content or ""
    result = parse_json_response(raw)

    return {
        "id": case["id"],
        "type": case["type"],
        "claim": case["claim"],
        "expected_verdict": case["expected_verdict"],
        "predicted_verdict": result["verdict"],
        "verdict_correct": (
            result["verdict"] == case["expected_verdict"]
        ),
        "expected_citations": case.get("expected_citations", []),
        "explanation": result["explanation"],
        "status": "completed",
    }


def main():
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    results = []
    errors = []

    print("\n========== REAL-MODEL EVALUATION ==========\n")
    print(f"Model: {citation_verifier.MODEL_NAME}\n")

    for case in dataset:
        try:
            result = evaluate_case(case)
            results.append(result)

            print(f"Case: {result['id']} ({result['type']})")
            print(f"Claim: {result['claim']}")
            print(f"Expected: {result['expected_verdict']}")
            print(f"Predicted: {result['predicted_verdict']}")
            print(f"Explanation: {result['explanation']}")
            print(
                "Result:",
                "PASS" if result["verdict_correct"] else "FAIL",
            )
            print("-" * 50)

        except Exception as exc:
            errors.append({
                "id": case["id"],
                "error": str(exc),
            })

            print(f"ERROR {case['id']}: {exc}")
            print("-" * 50)

    evaluated = len(results)
    correct = sum(
        result["verdict_correct"] for result in results
    )

    accuracy = correct / evaluated if evaluated else None

    report = {
        "model": citation_verifier.MODEL_NAME,
        "dataset": DATASET_PATH.name,
        "total_cases": len(dataset),
        "evaluated_cases": evaluated,
        "correct_verdicts": correct,
        "verdict_accuracy": accuracy,
        "errors": errors,
        "results": results,
    }

    RESULTS_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== SUMMARY ==========")
    print(f"Dataset cases: {len(dataset)}")
    print(f"Evaluated: {evaluated}")
    print(f"Correct verdicts: {correct}/{evaluated}")

    if accuracy is not None:
        print(f"Verdict accuracy: {accuracy:.1%}")
    else:
        print("Verdict accuracy: unavailable")

    print(f"Errors: {len(errors)}")
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
