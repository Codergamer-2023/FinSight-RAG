from backend.retrieval.keyword_retriever import KeywordRetriever
from backend.retrieval.retriever import Retriever


retriever = Retriever()

keyword_retriever = KeywordRetriever(
    retriever.client
)

question = "What was NVIDIA's total revenue in fiscal 2026?"

results = keyword_retriever.retrieve(
    question,
    top_k=5,
)

print(
    f"BM25 results: {len(results)}"
)

for rank, chunk in enumerate(results, start=1):
    print(
        f"\nRank {rank}"
        f"\nScore: {chunk.score:.4f}"
        f"\nDocument: {chunk.document}"
        f"\nPage: {chunk.page}"
        f"\nText: {chunk.text[:300]}"
    )