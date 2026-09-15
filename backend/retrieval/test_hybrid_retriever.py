from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.keyword_retriever import KeywordRetriever


def test_hybrid_retriever_returns_candidates(retriever):
    keyword_retriever = KeywordRetriever(
        retriever.client
    )

    hybrid_retriever = HybridRetriever(
        retriever,
        keyword_retriever,
    )

    results = hybrid_retriever.retrieve(
        "What was NVIDIA's total revenue in fiscal 2026?",
        top_k=10,
    )

    assert results
    assert len(results) <= 20

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