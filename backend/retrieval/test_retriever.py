def test_retriever_returns_results(retriever):
    results = retriever.retrieve(
        "What was NVIDIA's total revenue in fiscal 2026?",
        top_k=10,
        score_threshold=0.0,
    )

    assert results
    assert len(results) <= 10

    assert all(
        result.document
        for result in results
    )

    assert all(
        result.page > 0
        for result in results
    )

    assert all(
        result.text
        for result in results
    )