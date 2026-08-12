import logging 
from fastapi import FastAPI

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
