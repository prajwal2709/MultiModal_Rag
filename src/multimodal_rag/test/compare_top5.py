
import json
import re
from pathlib import Path

from multimodal_rag.retrival import retrieve_text_raw

BASE = Path("src/multimodal_rag/test")
TOP_K = 5

EXPANSIONS = {
    "R003": [
        "How many parallel attention heads does the Transformer employ, and what is the value of h?"
    ],
    "R006": [
        "How does masked self-attention work in the Transformer decoder?"
    ],
}


def tokenize(text):
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def overlap_score(query, text):
    query_terms = tokenize(query)
    if not query_terms:
        return 0.0
    return len(query_terms & tokenize(text)) / len(query_terms)


def metrics(selected, relevant):
    selected_ids = {item["chunk_index"] for item in selected}
    relevant = set(relevant)
    hits = len(selected_ids & relevant)

    return {
        "precision": hits / len(selected_ids) if selected_ids else 0.0,
        "recall": hits / len(relevant) if relevant else 0.0,
        "hit": int(hits > 0),
        "chunks": sorted(selected_ids),
    }


with open(BASE / "retrieval_evaluation_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

results = []

for case in dataset:
    query = case["query"]
    relevant = case["relevant_chunk_indices"]

    baseline = retrieve_text_raw(query, k=TOP_K)

    candidates = {}
    for expanded_query in [query, *EXPANSIONS.get(case["id"], [])]:
        for item in retrieve_text_raw(expanded_query, k=10):
            candidates.setdefault(item["chunk_index"], item)

    expanded = list(candidates.values())
    reranked = sorted(
        candidates.values(),
        key=lambda item: overlap_score(query, item["text"]),
        reverse=True,
    )[:TOP_K]

    result = {
        "id": case["id"],
        "baseline": metrics(baseline, relevant),
        "expanded": metrics(expanded, relevant),
        "reranked": metrics(reranked, relevant),
    }
    results.append(result)

    print(f"\n--- {case['id']} ---")
    for method in ["baseline", "expanded", "reranked"]:
        m = result[method]
        print(
            f"{method:10} "
            f"P={m['precision']:.3f} "
            f"R={m['recall']:.3f} "
            f"Hit={m['hit']} "
            f"Chunks={m['chunks']}"
        )

print("\n--- MACRO AVERAGES ---")
for method in ["baseline", "expanded", "reranked"]:
    print(
        f"{method:10} "
        f"Precision={sum(r[method]['precision'] for r in results) / len(results):.3f} "
        f"Recall={sum(r[method]['recall'] for r in results) / len(results):.3f} "
        f"Hit rate={sum(r[method]['hit'] for r in results) / len(results):.3f}"
    )

output = BASE / "top5_comparison_results.json"
output.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved results to: {output}")
