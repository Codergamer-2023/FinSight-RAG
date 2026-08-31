from backend.llm.generator import Generator
from backend.retrieval.retriever import Retriever


retriever = Retriever()
generator = Generator()

question = "What drove NVIDIA's revenue growth in fiscal 2026?"

chunks = retriever.retrieve(
    question,
    top_k=5,
)

answer = generator.generate(
    question,
    chunks,
)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)

print("\nSources:")
for chunk in chunks:
    print(f"- {chunk.document}, page {chunk.page}, score {chunk.score:.4f}")
