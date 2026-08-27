from pydantic import BaseModel

class RetrievedChunk(BaseModel):
    document: str
    page: int
    text: str
    score: float
