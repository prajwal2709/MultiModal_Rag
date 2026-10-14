"""Compare raw vector retrieval, neighbor expansion, and targeted expansion."""

import json
import math
from pathlib import Path

from multimodal_rag.retrival import retrieve_images, retrieve_text, retrieve_text_raw


TEST_DIR = Path(__file__).resolve().parent
DATASET_PATH = TEST_DIR / "retrieval_evaluation_dataset.json"
RESULTS_PATH = TEST_DIR / "retrieval_evaluation_results.json"
TOP_K = 3
IMAGE_K = 2
EXPANSION_QUERIES = {
    "R003": [
        "How many parallel attention heads does the Transformer employ, and what is the value of h?"
    ],
    "R006": ["How does masked self-attention work in the Transformer decoder?"],
}


def ranked_metrics(relevant_items, retrieved_items):
    """Binary graded metrics over unique ranked IDs; empty relevance is N/A."""
    relevant = set(relevant_items)
    ranked = list(dict.fromkeys(retrieved_items))
    if not relevant:
        return {
            # An explicit empty relevance set means all retrieved items are
            # false positives; keep precision useful for irrelevant-image cases.
            "precision": 0.0 if ranked else None,
            "recall": None,
            "mrr": None,
            "ndcg": None,
            "hit_rate": 0 if ranked else None,
            "relevant_found": 0,
            "relevant_total": 0,
            "unique_retrieved": len(ranked),
        }

    found = [item for item in ranked if item in relevant]
    reciprocal_rank = next(
        (1 / index for index, item in enumerate(ranked, start=1) if item in relevant),
        0.0,
    )
    dcg = sum(
        1 / math.log2(index + 1)
        for index, item in enumerate(ranked, start=1)
        if item in relevant
    )
    ideal = sum(
        1 / math.log2(index + 1)
        for index in range(1, min(len(relevant), len(ranked)) + 1)
    )
    return {
        "precision": round(len(found) / len(ranked), 4) if ranked else 0.0,
        "recall": round(len(found) / len(relevant), 4),
        "mrr": round(reciprocal_rank, 4),
        "ndcg": round(dcg / ideal, 4) if ideal else 0.0,
        "hit_rate": int(bool(found)),
        "relevant_found": len(found),
        "relevant_total": len(relevant),
        "unique_retrieved": len(ranked),
    }


def _identity(item):
    return (item.get("document_id", "unknown"), item["chunk_index"])


def evaluate_case(case):
    query = case["query"]
    baseline = retrieve_text_raw(query, k=TOP_K)
    improved = retrieve_text(query, k=TOP_K)

    expanded_by_id = {}
    queries = [query, *EXPANSION_QUERIES.get(case["id"], [])]
    for expanded_query in queries:
        for item in retrieve_text_raw(expanded_query, k=TOP_K):
            identity = _identity(item)
            if identity not in expanded_by_id:
                expanded_by_id[identity] = item
            elif expanded_query not in expanded_by_id[identity]["retrieved_by"]:
                expanded_by_id[identity]["retrieved_by"].append(expanded_query)
    expanded = list(expanded_by_id.values())

    relevant = set(case["relevant_chunk_indices"])
    baseline_ids = [_identity(item)[1] for item in baseline]
    improved_ids = [_identity(item)[1] for item in improved]
    expanded_ids = [_identity(item)[1] for item in expanded]
    images = retrieve_images(query, k=6)
    relevant_text_pages = {item["page"] for item in improved}
    for item in images:
        item["text_page_distance"] = min(
            abs(item["page"] - page) for page in relevant_text_pages
        ) if relevant_text_pages else None
        item["rank_score"] = item["distance"] + (
            0.035 * min(item["text_page_distance"], 2)
            if item["text_page_distance"] is not None else 0.0
        )
    images.sort(
        key=lambda item: (
            item["rank_score"],
            item["distance"],
        )
    )
    images = images[:IMAGE_K]
    image_ids = [item["image_id"] for item in images]

    return {
        "id": case["id"],
        "query": query,
        "baseline": {
            "retrieved_chunk_indices": baseline_ids,
            "context_size": len(baseline_ids),
            "metrics": ranked_metrics(relevant, baseline_ids),
        },
        "improved_with_neighbors": {
            "retrieved_chunk_indices": improved_ids,
            "context_size": len(improved_ids),
            "metrics": ranked_metrics(relevant, improved_ids),
        },
        "targeted_query_expansion": {
            "queries": queries,
            "retrieved_chunk_indices": expanded_ids,
            "retrieval_provenance": {
                f"{item.get('document_id', 'unknown')}:{item['chunk_index']}": item["retrieved_by"]
                for item in expanded
            },
            "context_size": len(expanded_ids),
            "metrics": ranked_metrics(relevant, expanded_ids),
        },
        "image_retrieval": {
            "retrieved_image_ids": image_ids,
            "retrieved_image_pages": [item["page"] for item in images],
            "same_page_as_retrieved_text": [
                item["page"] in relevant_text_pages for item in images
            ],
            "context_size": len(image_ids),
            "metrics": ranked_metrics(case["relevant_image_ids"], image_ids),
        },
    }


def _macro(results, section, metric):
    values = [
        item[section]["metrics"][metric]
        for item in results
        if item[section]["metrics"][metric] is not None
    ]
    return round(sum(values) / len(values), 4) if values else None


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    results, errors = [], []
    for case in dataset:
        try:
            result = evaluate_case(case)
            results.append(result)
            print(
                f"{case['id']}: "
                f"baseline R@{TOP_K}={result['baseline']['metrics']['recall']}, "
                f"neighbors R={result['improved_with_neighbors']['metrics']['recall']} "
                f"(context={result['improved_with_neighbors']['context_size']}), "
                f"expanded R={result['targeted_query_expansion']['metrics']['recall']} "
                f"(context={result['targeted_query_expansion']['context_size']})"
            )
        except Exception as error:
            errors.append({"id": case["id"], "error": str(error)})
            print(f"Error evaluating {case['id']}: {error}")

    sections = [
        "baseline",
        "improved_with_neighbors",
        "targeted_query_expansion",
        "image_retrieval",
    ]
    aggregates = {
        name: {
            metric: _macro(results, name, metric)
            for metric in ("precision", "recall", "mrr", "ndcg", "hit_rate")
        }
        for name in sections
    }
    report = {
        "dataset_size": len(dataset),
        "evaluated_cases": len(results),
        "failed_cases": errors,
        "top_k": TOP_K,
        "metric_definition": (
            "Unique document/chunk IDs and binary relevance. Precision uses "
            "unique context size; known empty relevance sets with returned items "
            "count as zero precision. Recall divides unique relevant retrieved "
            "IDs by unique relevant IDs. Recall, MRR, and nDCG macro-average over "
            "cases with relevant labels. "
            "MRR and nDCG use the returned ranking."
        ),
        "aggregate_metrics": aggregates,
        "results": results,
    }
    RESULTS_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Results saved to:", RESULTS_PATH)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
