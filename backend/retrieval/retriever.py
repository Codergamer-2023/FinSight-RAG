import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from backend.llm.cohere_client import CohereClient
from backend.retrieval.schemas import RetrievedChunk


load_dotenv()


COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "finsight_documents",
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

    return QdrantClient(
        path=QDRANT_PATH,
    )


class Retriever:
    def __init__(self):
        self.client = create_qdrant_client()
        self.cohere = CohereClient()

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
        score_threshold: float = 0.55,
    ) -> list[RetrievedChunk]:

        if not self.client.collection_exists(
            COLLECTION_NAME
        ):
            return []

        query_embedding = self.cohere.embed_query(
            question
        )

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
        )

        return [
            RetrievedChunk(
                document=result.payload["document"],
                page=result.payload["page"],
                text=result.payload["text"],
                score=result.score,
            )
            for result in results.points
            if result.score >= score_threshold
        ]
