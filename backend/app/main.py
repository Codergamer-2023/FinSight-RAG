import logging
import os
from fastapi import FastAPI, HTTPException

from backend.app.schemas import QueryRequest, QueryResponse, Source

from backend.retrieval.retriever import Retriever

from backend.retrieval.reranker import Reranker

from backend.llm.generator import Generator

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
app = FastAPI(
    title = "FinSight API",
    version = "0.1.0",
)

retriever = Retriever()
generator = Generator()
reranker = Reranker()

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

@app.get("/health")
async def health_check() -> dict[str, str]:
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "service": "FinSight API",
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    logger.info("Query received")
    try:
        chunks = retriever.retrieve(
            request.question,
            top_k=RETRIEVAL_TOP_K,
        )
        if not chunks:
            logger.info("No relevant documents found")

            return QueryResponse(
                answer=(
                    "The information is not available in "
                    "the provided documents."
                ),
                sources=[],
            )
        chunks = reranker.rerank(
            request.question,
            chunks,
            top_k=RERANK_TOP_K,
        )
        if not chunks:
            logger.info("No relevant chunks after reranking")

            return QueryResponse(
                answer=(
                    "The information is not available in "
                    "the provided documents."
                ),
                sources=[],
            )
        answer = generator.generate(
            request.question,
            chunks,
        )
        sources = []
        seen_sources = set()

        for chunk in chunks:
            source_key = (chunk.document, chunk.page)

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            sources.append(
                Source(
                    document=chunk.document,
                    page=chunk.page,
                )
            )
        return QueryResponse(
            answer=answer,
            sources=sources,
        )
    except Exception:
        logger.exception("Query processing failed")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the query.",
        )
