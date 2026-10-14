
import json

from multimodal_rag import citation_verifier


TEST_CASES = [
    {
        "name": "supported_claim",
        "evidence": (
            "The encoder consists of a stack of N = 6 identical layers."
        ),
        "claim": "The encoder consists of six identical layers.",
        "expected": "supported",
    },
    {
        "name": "contradicted_claim",
        "evidence": (
            "The encoder consists of a stack of N = 6 identical layers."
        ),
        "claim": "The encoder consists of eight identical layers.",
        "expected": "contradicted",
    },
    {
        "name": "insufficient_evidence",
        "evidence": (
            "The encoder consists of a stack of N = 6 identical layers."
        ),
        "claim": "The encoder uses eight attention heads.",
        "expected": "insufficient_evidence",
    },
]


def evaluate_case(case):
    """Ask the configured model to classify one claim against evidence."""
    prompt = f"""
You are evaluating a factual claim against supplied evidence.

Evidence:
{case["evidence"]}

Claim:
{case["claim"]}

Choose exactly one verdict:
- supported: the evidence supports the claim.
- contradicted: the evidence directly conflicts with the claim.
- insufficient_evidence: the evidence does not establish whether the claim is true.

Do not use outside knowledge.

Return only valid JSON with these fields:
{{
  "verdict": "supported, contradicted, or insufficient_evidence",
  "explanation": "brief explanation"
}}
"""

    response = citation_verifier.client.chat.completions.create(
        model=citation_verifier.MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )

    raw = (response.choices[0].message.content or "").strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].strip()

    result = json.loads(raw)

    allowed_verdicts = {
        "supported",
        "contradicted",
        "insufficient_evidence",
    }

    if result.get("verdict") not in allowed_verdicts:
        raise ValueError(
            f"Unexpected verdict for {case['name']}: "
            f"{result.get('verdict')}"
        )

    return result


def main():
    correct = 0

    print("\n========== REAL-MODEL EVALUATION ==========\n")

    for case in TEST_CASES:
        try:
            result = evaluate_case(case)
            passed = result["verdict"] == case["expected"]
            correct += int(passed)

            print(f"Test: {case['name']}")
            print(f"Claim: {case['claim']}")
            print(f"Expected: {case['expected']}")
            print(f"Actual: {result['verdict']}")
            print(f"Explanation: {result.get('explanation', '')}")
            print(f"Result: {'PASS' if passed else 'FAIL'}")
            print("-" * 50)

        except Exception as exc:
            print(f"Test: {case['name']}")
            print(f"ERROR: {exc}")
            print("-" * 50)

    total = len(TEST_CASES)

    print("\n========== SUMMARY ==========")
    print(f"Correct verdicts: {correct}/{total}")
    print(f"Accuracy: {correct / total:.1%}")


if __name__ == "__main__":
    main()
