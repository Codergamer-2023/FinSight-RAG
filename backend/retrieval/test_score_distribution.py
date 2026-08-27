from backend.retrieval.retriever import Retriever

retriever = Retriever()

questions = [
    "What drove NVIDIA's revenue growth in fiscal 2026?",
    "What was NVIDIA's total revenue in fiscal 2026?",
    "How much did NVIDIA's Data Center revenue grow in fiscal 2026?",
    "What risks could affect NVIDIA's future revenue?",
    "What was NVIDIA's net income in fiscal 2026?",
]

for question in questions:
    print(f"\nQUESTION: {question}")

    results = retriever.retrieve(question, top_k=5)

    for i, result in enumerate(results, start=1):
        print(
            f"  {i}. score={result.score:.4f}, "
            f"page={result.page}"
        )
