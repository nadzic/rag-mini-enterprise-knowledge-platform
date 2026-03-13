from fastapi import APIRouter

from schemas.rag import RagQueryRequest, RagQueryResponse

router = APIRouter(tags=["rag"])

@router.post("/rag/query", response_model=RagQueryResponse)
async def reg_query(request: RagQueryRequest) -> RagQueryResponse:
  return RagQueryResponse(
    answer="This is a test answer",
    sources=["source1", "source2"],
    num_contexts=10
  )