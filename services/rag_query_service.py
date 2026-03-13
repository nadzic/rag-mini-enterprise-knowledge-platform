from __future__ import annotations

from typing import TypedDict

from openai import AsyncOpenAI

from rag_types import RAGQueryResult
from services.openai_api_key import resolve_openai_api_key
from services.query_retrieval import search_contexts

_async_openai_client: AsyncOpenAI | None = None


class RagQueryResultDict(TypedDict):
    answer: str
    sources: list[str]
    num_contexts: int


def _get_async_openai_client() -> AsyncOpenAI:
    global _async_openai_client
    if _async_openai_client is not None:
        return _async_openai_client
    _async_openai_client = AsyncOpenAI(api_key=resolve_openai_api_key())
    return _async_openai_client


def _build_user_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"- {context}" for context in contexts)
    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )


async def run_rag_query(question: str, top_k: int) -> RagQueryResultDict:
    found = search_contexts(question, top_k)
    user_content = _build_user_prompt(question, found.contexts)

    response = await _get_async_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": "You answer questions based on the provided context.",
            },
            {"role": "user", "content": user_content},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()

    result = RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
    )
    return {
        "answer": result.answer,
        "sources": result.sources,
        "num_contexts": result.num_contexts,
    }
