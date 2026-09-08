from backend.retrieval.keyword_retriever import KeywordRetriever
from backend.retrieval.retriever import Retriever
from backend.retrieval.schemas import RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        semantic_retriever: Retriever,
        keyword_retriever: KeywordRetriever,
    ):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        semantic_results = (
            self.semantic_retriever.retrieve(
                question,
                top_k=top_k,
                score_threshold=0.0,
            )
        )
        keyword_results = self.keyword_retriever.retrieve(
            question,
            top_k=top_k,
        )

        merged = {}

        for chunk in semantic_results:
            key = (
                chunk.document,
                chunk.page,
                chunk.text,
            )

            merged[key] = chunk

        for chunk in keyword_results:
            key = (
                chunk.document,
                chunk.page,
                chunk.text,
            )

            if key not in merged:
                merged[key] = chunk

        return list(merged.values())