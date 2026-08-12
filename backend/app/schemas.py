from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's financial research question.",
    )

class Source(BaseModel):
    document: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

