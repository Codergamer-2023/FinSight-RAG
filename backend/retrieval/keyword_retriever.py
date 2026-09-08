import re

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from backend.retrieval.schemas import RetrievedChunk


COLLECTION_NAME = "finsight_documents"


class KeywordRetriever:
    def __init__(
        self,
        client: QdrantClient,
    ):
        self.client = client
        self.chunks: list[RetrievedChunk] = []
        self.bm25: BM25Okapi | None = None

        self._load_from_qdrant()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def _load_from_qdrant(self) -> None:
        chunks = []

        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                offset=offset,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload

                if not payload:
                    continue

                chunks.append(
                    RetrievedChunk(
                        document=payload["document"],
                        page=payload["page"],
                        text=payload["text"],
                        score=0.0,
                    )
                )

            if next_offset is None:
                break

            offset = next_offset

        self.chunks = chunks
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self.chunks:
            self.bm25 = None
            return

        tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    def add_chunks(
        self,
        chunks: list[RetrievedChunk],
    ) -> None:

        existing = {
            (
                chunk.document,
                chunk.page,
                chunk.text,
            )
            for chunk in self.chunks
        }

        for chunk in chunks:
            key = (
                chunk.document,
                chunk.page,
                chunk.text,
            )

            if key not in existing:
                self.chunks.append(chunk)
                existing.add(key)

        self._rebuild_index()

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:

        if not self.chunks or self.bm25 is None:
            return []

        tokenized_query = self._tokenize(question)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.chunks[index].model_copy(
                update={
                    "score": score,
                }
            )

            results.append(chunk)

        return results