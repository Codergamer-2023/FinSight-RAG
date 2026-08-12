import logging 
from fastapi import FastAPI

from backend.app.schemas import QueryRequest


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
app = FastAPI(
    title = "FinSight API",
    version = "0.1.0",
)
@app.get("/health")
async def health_check() -> dict[str, str]:
    logger.info("Health check requested")
    return {"status": "healthy"}

@app.post("/api/v1/query")
async def query(request: QueryRequest) -> dict[str, str]:
    logger.info("Query received")

    return {
        "message": "Query accepted",
        "question": request.question,
    }