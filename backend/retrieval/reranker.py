from sentence_transformers import CrossEncoder
from backend.retrieval.schemas import RetrievedChunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(
            self,
            question : str,
            chunks : list[RetrievedChunk],
            top_k : int = 5
    ) -> list[RetrievedChunk]:

        if not chunks:
            return []

        pairs = [
            [question, chunk.text]
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)
        ranked_chunks = [
            chunk.model_copy(
                update={"rerank_score": float(score)}
            )
            for chunk, score in zip(chunks, scores)
        ]
        ranked_chunks.sort(
            key=lambda chunk: chunk.rerank_score,
            reverse=True,
        )
        return ranked_chunks[:top_k]
