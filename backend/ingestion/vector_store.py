import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "finsight_documents",
)

VECTOR_SIZE = int(
    os.getenv(
        "COHERE_EMBED_DIMENSION",
        "1024",
    )
)

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "data/qdrant",
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
)


def create_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    Path(QDRANT_PATH).mkdir(
        parents=True,
        exist_ok=True,
    )

    return QdrantClient(
        path=QDRANT_PATH,
    )


def create_vector_store(
    chunks: list[Document],
    embeddings: list[list[float]],
) -> QdrantClient:

    client = create_qdrant_client()

    if not client.collection_exists(
        COLLECTION_NAME
    ):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    points = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
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


def add_to_vector_store(
    chunks: list[Document],
    embeddings: list[list[float]],
    client: QdrantClient,
) -> QdrantClient:

    if not client.collection_exists(
        COLLECTION_NAME
    ):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    points = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                (
                    f"{chunk.metadata['document']}:"
                    f"{chunk.metadata['page']}:"
                    f"{chunk.page_content}"
                ),
            )
        )

        points.append(
            PointStruct(
                id=point_id,
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

    client = create_vector_store(
        chunks,
        embeddings,
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    print(f"Chunks: {len(chunks)}")
    print(
        f"Vectors: {collection.points_count}"
    )