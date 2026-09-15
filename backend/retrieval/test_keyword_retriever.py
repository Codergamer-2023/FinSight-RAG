from backend.retrieval.keyword_retriever import KeywordRetriever


def test_keyword_retriever_returns_results(retriever):
    keyword_retriever = KeywordRetriever(
        retriever.client
    )

    results = keyword_retriever.retrieve(
        "NVIDIA total revenue fiscal 2026",
        top_k=10,
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