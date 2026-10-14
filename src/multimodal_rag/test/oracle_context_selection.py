
import json
from pathlib import Path

BASE = Path("src/multimodal_rag/test")

with open(BASE / "retrieval_evaluation_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

with open(BASE / "top5_comparison_results.json", encoding="utf-8") as f:
    results = json.load(f)

print("CASE | CANDIDATES | RELEVANT FOUND | MINIMUM ORACLE CONTEXT")

for result in results:
    case = next(c for c in dataset if c["id"] == result["id"])
    candidates = set(result["expanded"]["chunks"])
    relevant = set(case["relevant_chunk_indices"])

    found = candidates & relevant
    missing = relevant - candidates

    print(
        f"{result['id']} | "
        f"{len(candidates)} | "
        f"{len(found)}/{len(relevant)} | "
        f"{len(found)}"
    )

    if missing:
        print(f"  Missing relevant chunks: {sorted(missing)}")

    if found:
        print(f"  Oracle-selected chunks: {sorted(found)}")

print(
    "\nNote: Oracle selection uses ground-truth labels and is "
    "only an evaluation upper bound, not a deployable retriever."
)
