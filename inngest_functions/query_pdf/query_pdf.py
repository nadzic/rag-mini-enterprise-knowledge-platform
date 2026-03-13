import inngest
from typing import Protocol, cast

from services.rag_query_service import RagQueryResultDict, run_rag_query
from services.langfuse_client import flush_langfuse

from ..client import inngest_client
from .query_pdf_finalize_failure import finalize_failure
from .query_pdf_parse_inputs import parse_query_inputs
from .query_pdf_start_span import start_root_span


class _SpanLike(Protocol):
    def start_span(self, *, name: str) -> "_SpanLike": ...
    def update(self, *, output: RagQueryResultDict) -> None: ...
    def end(self) -> None: ...


@inngest_client.create_function(
    fn_id="RAG: Query",
    trigger=inngest.TriggerEvent(event="rag/query"),
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> RagQueryResultDict:
    """Orchestrate retrieval + generation and return a JSON-safe query result."""
    question, top_k = parse_query_inputs(ctx)
    root_span = cast(_SpanLike | None, start_root_span(ctx, question, top_k))

    execution_span = root_span.start_span(name="rag-query-core") if root_span else None
    try:
        async def _run_query_step() -> RagQueryResultDict:
            return await run_rag_query(question, top_k)

        result_dict = await ctx.step.run("run-rag-query", _run_query_step)
        if execution_span is not None:
            execution_span.update(output=result_dict)
    except Exception as exc:
        finalize_failure(root_span, execution_span, exc)
        raise
    else:
        if execution_span is not None:
            execution_span.end()
    if root_span is not None:
        root_span.update(output=result_dict)
        root_span.end()
    flush_langfuse()
    # Inngest expects JSON-serializable return payloads.
    return result_dict
