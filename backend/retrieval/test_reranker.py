from backend.retrieval.retriever import Retriever
from backend.retrieval.reranker import Reranker


retriever = Retriever()
reranker = Reranker()

question = "What was NVIDIA's total revenue in fiscal 2026?"

chunks = retriever.retrieve(
    question,
    top_k=10,
)

reranked_chunks = reranker.rerank(
    question,
    chunks,
    top_k=5,
)
reranked_chunks = reranker.rerank(
    question,
    chunks,
    top_k=5,
)

print(f"Question: {question}")
print(f"\nRetrieved candidates: {len(chunks)}")
print(f"Reranked results: {len(reranked_chunks)}")

for i, chunk in enumerate(reranked_chunks, start=1):
    print(f"\n--- Reranked Result {i} ---")
    print(f"Document: {chunk.document}")
    print(f"Page: {chunk.page}")
    print(f"Qdrant Score: {chunk.score:.4f}")
    print(f"Rerank Score: {chunk.rerank_score:.4f}")
    print(f"Text: {chunk.text[:700]}")
