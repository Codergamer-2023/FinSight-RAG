from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_core.documents import Document

COLLECTION_NAME = "finsight_documents"
VECTOR_SIZE = 384
QDRANT_PATH = "data/qdrant"

def create_vector_store(
    chunks: list[Document],
    embeddings: list[list[float]],
) -> QdrantClient:
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=QDRANT_PATH)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    points = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=index,
                vector=embedding,
                payload={
                    "text": chunk.page_content,
                    "document": chunk.metadata["document"],
                    "page": chunk.metadata["page"],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return client

if __name__ == "__main__":
    from backend.ingestion.load_pdf import load_pdf
    from backend.ingestion.chunk_documents import chunk_documents
    from backend.ingestion.embed_documents import embed_documents

    documents = load_pdf()
    chunks = chunk_documents(documents)
    embeddings = embed_documents(chunks)

    client = create_vector_store(chunks, embeddings)

    collection = client.get_collection(COLLECTION_NAME)

    print(f"Chunks: {len(chunks)}")
    print(f"Vectors: {collection.points_count}")