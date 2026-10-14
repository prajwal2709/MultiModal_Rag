
import json
from pathlib import Path

from multimodal_rag.retrival import retrieve_text_raw

DATASET_PATH = Path(
    "src/multimodal_rag/test/retrieval_evaluation_dataset.json"
)

# The original query and a targeted follow-up query.
# We use the follow-up only for the multi-head-attention case.

EXPANSION_QUERIES = {
    "R003": [
        "How many parallel attention heads does the Transformer employ, and what is the value of h?"
    ],
    "R006": [
        "How does masked self-attention work in the Transformer decoder?"
    ],
}


K = 10


def retrieve_baseline(query):
    results = retrieve_text_raw(query, k=K)
    return {item["chunk_index"]: item for item in results}


def retrieve_expanded(case):
    queries = [case["query"]]
    queries.extend(EXPANSION_QUERIES.get(case["id"], []))

    merged = {}
    for query in queries:
        for item in retrieve_text_raw(query, k=K):
            merged.setdefault(item["chunk_index"], item)

    return merged


def calculate_metrics(retrieved_indices, relevant_indices):
    retrieved = set(retrieved_indices)
    relevant = set(relevant_indices)

    if not relevant:
        return {
            "precision": None,
            "recall": None,
            "hit": None,
        }

    true_positives = len(retrieved & relevant)

    return {
        "precision": (
            true_positives / len(retrieved) if retrieved else 0.0
        ),
        "recall": true_positives / len(relevant),
        "hit": int(true_positives > 0),
    }


def main():
    with DATASET_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)

    results = []

    for case in cases:
        relevant = case["relevant_chunk_indices"]

        baseline = retrieve_baseline(case["query"])
        expanded = retrieve_expanded(case)

        baseline_metrics = calculate_metrics(baseline, relevant)
        expanded_metrics = calculate_metrics(expanded, relevant)

        result = {
            "id": case["id"],
            "query": case["query"],
            "baseline": {
                **baseline_metrics,
                "retrieved_chunks": list(baseline),
                "context_count": len(baseline),
            },
            "expanded": {
                **expanded_metrics,
                "retrieved_chunks": list(expanded),
                "context_count": len(expanded),
            },
        }
        results.append(result)

        print(f"\n--- {case['id']} ---")
        print(f"Query: {case['query']}")
        print(
            f"Baseline: recall={baseline_metrics['recall']:.3f}, "
            f"precision={baseline_metrics['precision']:.3f}, "
            f"context={len(baseline)}"
        )
        print(
            f"Expanded: recall={expanded_metrics['recall']:.3f}, "
            f"precision={expanded_metrics['precision']:.3f}, "
            f"context={len(expanded)}"
        )
        print(
            "Retrieved relevant chunks:",
            sorted(set(expanded) & set(relevant)),
        )

    output_path = DATASET_PATH.with_name(
        "query_expansion_comparison.json"
    )
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
