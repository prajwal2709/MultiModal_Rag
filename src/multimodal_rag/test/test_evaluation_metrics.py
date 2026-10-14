from multimodal_rag.test.evaluate_retrieval import ranked_metrics


def test_ranked_metrics_deduplicate_repeated_chunk_ids():
    metrics = ranked_metrics(["a", "b"], ["x", "a", "a"])

    assert metrics["unique_retrieved"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["mrr"] == 0.5


def test_ranked_metrics_marks_empty_relevance_not_applicable():
    metrics = ranked_metrics([], ["image-1"])

    assert metrics["precision"] == 0.0
    assert metrics["recall"] is None
    assert metrics["ndcg"] is None
