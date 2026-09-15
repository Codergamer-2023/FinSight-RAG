def test_retrieval_score_distribution(retriever):
    results = retriever.retrieve(
        "What was NVIDIA's total revenue in fiscal 2026?",
        top_k=10,
        score_threshold=0.0,
    )

    assert results

    scores = [
        result.score
        for result in results
    ]

    assert all(
        0.0 <= score <= 1.0
        for score in scores
    )