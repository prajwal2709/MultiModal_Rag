
import json
from pathlib import Path

base = Path("src/multimodal_rag/test")

with open(base / "query_expansion_comparison.json", encoding="utf-8") as f:
    results = json.load(f)

with open(base / "retrieval_evaluation_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

ground_truth = {
    case["id"]: set(case["relevant_chunk_indices"])
    for case in dataset
}

print(
    f"{'CASE':<8} {'BASELINE':>10} {'EXPANDED':>10} "
    f"{'BASE IRR.':>10} {'EXP IRR.':>10}"
)

for result in results:
    case_id = result["id"]
    relevant = ground_truth[case_id]

    baseline = result["baseline"]["retrieved_chunks"]
    expanded = result["expanded"]["retrieved_chunks"]

    baseline_irrelevant = len(set(baseline) - relevant)
    expanded_irrelevant = len(set(expanded) - relevant)

    print(
        f"{case_id:<8} "
        f"{len(baseline):>10} "
        f"{len(expanded):>10} "
        f"{baseline_irrelevant:>10} "
        f"{expanded_irrelevant:>10}"
    )

print("\nTotals:")
print("Baseline chunks:", sum(
    r["baseline"]["context_count"] for r in results
))
print("Expanded chunks:", sum(
    r["expanded"]["context_count"] for r in results
))
