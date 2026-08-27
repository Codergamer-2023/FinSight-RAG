from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from backend.retrieval.schemas import RetrievedChunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "finsight_documents"
QDRANT_PATH = "data/qdrant"

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = QdrantClient(path = QDRANT_PATH)

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.model.encode(
            question,
            convert_to_numpy=True,
        ).tolist()

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
        ]
