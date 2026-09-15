from backend.retrieval.reranker import Reranker


def test_reranker_returns_ranked_results(retriever):
    chunks = retriever.retrieve(
        "What was NVIDIA's total revenue in fiscal 2026?",
        top_k=10,
        score_threshold=0.0,
    )

    assert chunks

    reranker = Reranker()

    results = reranker.rerank(
        "What was NVIDIA's total revenue in fiscal 2026?",
        chunks,
        top_k=5,
    )

    assert results
    assert len(results) <= 5

    assert all(
        chunk.rerank_score is not None
        for chunk in results
    )

    scores = [
        chunk.rerank_score
        for chunk in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )