from backend.llm.cohere_client import CohereClient

from backend.retrieval.schemas import RetrievedChunk


class Reranker:
    def __init__(self):
        self.cohere = CohereClient()

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:

        if not chunks:
            return []

        documents = [
            chunk.text
            for chunk in chunks
        ]

        results = self.cohere.rerank(
            query=question,
            documents=documents,
            top_n=min(top_k, len(documents)),
        )

        ranked_chunks = []

        for result in results:
            chunk = chunks[result.index]

            ranked_chunks.append(
                chunk.model_copy(
                    update={
                        "rerank_score": float(
                            result.relevance_score
                        )
                    }
                )
            )

        return ranked_chunks