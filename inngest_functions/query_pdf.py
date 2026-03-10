import inngest
from inngest.experimental import ai
from typing import Any

from rag_types import RAGQueryResult, RAGSearchResult
from services import QdrantVectorStore, embed_texts
from services.langfuse_client import flush_langfuse, get_langfuse_client

from .client import inngest_client
from .env import get_openai_api_key


@inngest_client.create_function(
    fn_id="RAG: Query",
    trigger=inngest.TriggerEvent(event="rag/query"),
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> RAGQueryResult:
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantVectorStore()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = str(ctx.event.data["question"])
    top_k_raw = ctx.event.data.get("top_k", 5)
    top_k = int(top_k_raw) if isinstance(top_k_raw, (int, float, str)) else 5

    langfuse = get_langfuse_client()
    root_span = None
    if langfuse is not None:
        root_span = langfuse.start_span(
            name="rag.query",
            input={"question": question, "top_k": top_k},
            metadata={
                "event_name": "rag/query",
                "event_id": str(getattr(ctx.event, "id", "")),
            },
        )
        user_id = ctx.event.data.get("user_id")
        session_id = ctx.event.data.get("session_id")
        langfuse.update_current_trace(
            user_id=str(user_id) if user_id else None,
            session_id=str(session_id) if session_id else None,
        )

    search_span = root_span.start_span(name="embed-and-search") if root_span else None
    try:
        async def _search_step() -> RAGSearchResult:
            return _search(question, top_k)

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
        if search_span is not None:
            search_span.update(level="ERROR", status_message=str(exc))
            search_span.end()
        if root_span is not None:
            root_span.update(level="ERROR", status_message=str(exc))
            root_span.end()
        flush_langfuse()
        raise
    else:
        if search_span is not None:
            search_span.end()

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=get_openai_api_key(),
        model="gpt-4o-mini",
    )

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
        res = await ctx.step.ai.infer(
            "llm-answer",
            adapter=adapter,
            body={
                "max_tokens": 1024,
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "system",
                        "content": "You answer questions based on the provided context.",
                    },
                    {"role": "user", "content": user_content},
                ],
            },
        )
    except Exception as exc:
        if generation is not None:
            generation.update(level="ERROR", status_message=str(exc))
            generation.end()
        if root_span is not None:
            root_span.update(level="ERROR", status_message=str(exc))
            root_span.end()
        flush_langfuse()
        raise

    res_data = res if isinstance(res, dict) else {}
    raw_choices = res_data.get("choices", [])
    choices = raw_choices if isinstance(raw_choices, list) else []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    answer = str(message.get("content", "")).strip()
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
    return result