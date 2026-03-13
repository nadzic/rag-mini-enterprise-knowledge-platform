from fastapi import APIRouter

from schemas.rag import RagQueryRequest, RagQueryResponse
from services.rag_query_service import run_rag_query

router = APIRouter(tags=["rag"])

@router.post("/rag/query", response_model=RagQueryResponse)
async def reg_query(request: RagQueryRequest) -> RagQueryResponse:
    result = await run_rag_query(request.question, request.top_k)
    return RagQueryResponse(**result)