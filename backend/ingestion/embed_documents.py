from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_documents(documents: list[Document]) -> list[list[float]]:
    model = SentenceTransformer(MODEL_NAME)

    texts = [document.page_content for document in documents]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
    )

    return embeddings.tolist()

if __name__ == "__main__":
    from backend.ingestion.load_pdf import load_pdf
    from backend.ingestion.chunk_documents import chunk_documents

    documents = load_pdf()
    chunks = chunk_documents(documents)

    embeddings = embed_documents(chunks)

    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {len(embeddings)}")
    print(f"Dimensions: {len(embeddings[0])}")
    print(f"First 5 values: {embeddings[0][:5]}")
