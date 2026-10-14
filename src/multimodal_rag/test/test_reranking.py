
import json
import re
from pathlib import Path

from multimodal_rag.retrival import retrieve_text_raw

BASE = Path("src/multimodal_rag/test")
TOP_K = 5

with open(BASE / "retrieval_evaluation_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

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


def score(query, text):
    query_terms = tokenize(query)
    text_terms = tokenize(text)

    if not query_terms:
        return 0.0

    return len(query_terms & text_terms) / len(query_terms)


def main():
    for case in dataset:
        queries = [case["query"], *EXPANSIONS.get(case["id"], [])]
        candidates = {}

        for query in queries:
            for item in retrieve_text_raw(query, k=10):
                key = (item.get("document_id", "unknown"), item["chunk_index"])
                candidates.setdefault(key, item)

        ranked = sorted(
            candidates.values(),
            key=lambda item: score(case["query"], item["text"]),
            reverse=True,
        )

        selected = ranked[:TOP_K]
        selected_indices = {item["chunk_index"] for item in selected}
        relevant = set(case["relevant_chunk_indices"])
        found = selected_indices & relevant

        print(f"\n--- {case['id']} ---")
        print(f"Candidate count: {len(candidates)}")
        print(f"Selected chunks: {[item['chunk_index'] for item in selected]}")
        print(f"Relevant chunks found: {sorted(found)}")
        print(f"Recall@{TOP_K}: {len(found) / len(relevant):.3f}")


if __name__ == "__main__":
    main()
