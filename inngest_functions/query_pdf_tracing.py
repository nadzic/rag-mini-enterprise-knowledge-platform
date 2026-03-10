from typing import Any

import inngest

from services.langfuse_client import flush_langfuse, get_langfuse_client


def parse_query_inputs(ctx: inngest.Context) -> tuple[str, int]:
    """Read and normalize question/top_k from the incoming query event."""
    question = str(ctx.event.data["question"])
    top_k_raw = ctx.event.data.get("top_k", 5)
    top_k = int(top_k_raw) if isinstance(top_k_raw, (int, float, str)) else 5
    return question, top_k


def start_root_span(ctx: inngest.Context, question: str, top_k: int) -> Any:
    """Create the root Langfuse span for query execution when configured."""
    langfuse = get_langfuse_client()
    if langfuse is None:
        return None
    return langfuse.start_span(
        name="rag.query",
        input={"question": question, "top_k": top_k},
        metadata={
            "event_name": "rag/query",
            "event_id": str(getattr(ctx.event, "id", "")),
            "user_id": str(ctx.event.data.get("user_id", "")),
            "session_id": str(ctx.event.data.get("session_id", "")),
        },
    )


def finalize_failure(root_span: Any, step_span: Any, exc: Exception) -> None:
    """Mark spans as failed and flush Langfuse so error traces are persisted."""
    if step_span is not None:
        step_span.update(level="ERROR", status_message=str(exc))
        step_span.end()
    if root_span is not None:
        root_span.update(level="ERROR", status_message=str(exc))
        root_span.end()
    flush_langfuse()
