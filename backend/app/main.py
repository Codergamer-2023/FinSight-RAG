import logging 
from fastapi import FastAPI

from backend.app.schemas import QueryRequest, QueryResponse, Source

from backend.retrieval.retriever import Retriever

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

@app.get("/health")
async def health_check() -> dict[str, str]:
    logger.info("Health check requested")
    return {"status": "healthy"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    logger.info("Query received")

    chunks = retriever.retrieve(
        request.question,
        top_k=5,
    )

    sources = [
        Source(
            document=chunk.document,
            page=chunk.page,
        )
        for chunk in chunks
    ]

    return QueryResponse(
        answer="Retrieval successful",
        sources=sources,
    )
