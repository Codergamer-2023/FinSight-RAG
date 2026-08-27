from backend.retrieval.retriever import Retriever

retriever = Retriever()
question = "What drove NVIDIA's revenue growth in fiscal 2026?"

results = retriever.retrieve(
    question,
    top_k=5,
)

print(f"Question: {question}")
print(f"Results: {len(results)}")

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Score: {result.score:.4f}")
    print(f"Document: {result.document}")
    print(f"Page: {result.page}")
    print(f"Text: {result.text[:700]}")
