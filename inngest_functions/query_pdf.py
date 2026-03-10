import inngest

from rag_types import RAGQueryResult, RAGSearchResult
from services.langfuse_client import flush_langfuse

from .client import inngest_client
from .query_pdf_llm import build_user_prompt, extract_answer, infer_answer
from .query_pdf_search import search_contexts
from .query_pdf_tracing import finalize_failure, parse_query_inputs, start_root_span


@inngest_client.create_function(
    fn_id="RAG: Query",
    trigger=inngest.TriggerEvent(event="rag/query"),
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> dict:
    """Orchestrate retrieval + generation and return a JSON-safe query result."""
    question, top_k = parse_query_inputs(ctx)
    root_span = start_root_span(ctx, question, top_k)

    search_span = root_span.start_span(name="embed-and-search") if root_span else None
    try:
        async def _search_step() -> RAGSearchResult:
            return search_contexts(question, top_k)

        found = await ctx.step.run(
            "embed-and-search",
            _search_step,
            output_type=RAGSearchResult,
        )
        if search_span is not None:
            search_span.update(
                output={"num_contexts": len(found.contexts), "num_sources": len(found.sources)}
            )
    except Exception as exc:
        finalize_failure(root_span, search_span, exc)
        raise
    else:
        if search_span is not None:
            search_span.end()

    user_content = build_user_prompt(question, found.contexts)

    generation = (
        root_span.start_generation(
            name="llm-answer",
            model="gpt-4o-mini",
            input={"question": question, "num_contexts": len(found.contexts)},
        )
        if root_span
        else None
    )
    try:
        res = await infer_answer(ctx, user_content)
    except Exception as exc:
        if generation is not None:
            generation.update(level="ERROR", status_message=str(exc))
        finalize_failure(root_span, generation, exc)
        raise

    answer = extract_answer(res)
    result = RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
    )

    if generation is not None:
        generation.update(
            output={"answer_preview": answer[:500], "num_sources": len(found.sources)}
        )
        generation.end()
    if root_span is not None:
        root_span.update(output=result.model_dump())
        root_span.end()
    flush_langfuse()
    # Inngest expects JSON-serializable return payloads.
    return result.model_dump()