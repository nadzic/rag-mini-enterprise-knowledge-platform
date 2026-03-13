from pydantic import BaseModel, Field

class RagQueryRequest(BaseModel):
  question: str = Field(..., min_length=1, description="User question to the RAG system")
  top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")

class RagQueryResponse(BaseModel):
  answer: str
  sources: list[str]
  num_contexts: int