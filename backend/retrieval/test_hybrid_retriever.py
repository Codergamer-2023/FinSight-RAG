from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.keyword_retriever import KeywordRetriever
from backend.retrieval.reranker import Reranker
from backend.retrieval.retriever import Retriever


semantic_retriever = Retriever()

keyword_retriever = KeywordRetriever(
    semantic_retriever.client
)

hybrid_retriever = HybridRetriever(
    semantic_retriever,
    keyword_retriever,
)

reranker = Reranker()


questions = [
    "What was NVIDIA's total revenue in fiscal 2026?",
    "What was Acme Technologies total revenue in fiscal 2026?",
]


for question in questions:

    print("\n" + "=" * 80)
    print(f"Question: {question}")
    print("=" * 80)

    hybrid_results = hybrid_retriever.retrieve(
        question,
        top_k=10,
    )

    print(
        f"\nHybrid candidates: "
        f"{len(hybrid_results)}"
    )

    reranked_results = reranker.rerank(
        question,
        hybrid_results,
        top_k=5,
    )

    print(
        f"Reranked results: "
        f"{len(reranked_results)}"
    )

    for rank, chunk in enumerate(
        reranked_results,
        start=1,
    ):

        print(
            f"\nRank {rank}"
            f"\nDocument: {chunk.document}"
            f"\nPage: {chunk.page}"
            f"\nSemantic/BM25 score: "
            f"{chunk.score:.4f}"
            f"\nRerank score: "
            f"{chunk.rerank_score:.4f}"
            f"\nText: {chunk.text[:300]}"
        )