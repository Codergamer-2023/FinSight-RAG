from pydantic import BaseModel

class RetrievedChunk(BaseModel):
    document: str
    page: int
    text: str
    score: float
    rerank_score : float | None = None