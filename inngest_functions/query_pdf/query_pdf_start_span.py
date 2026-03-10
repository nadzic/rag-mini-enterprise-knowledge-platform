import hashlib
from typing import Any, cast

import inngest

from services.langfuse_client import get_langfuse_client


def start_root_span(ctx: inngest.Context, question: str, top_k: int) -> Any:
    """Create the root Langfuse span for query execution when configured."""
    langfuse = get_langfuse_client()
    if langfuse is None:
        return None
    event_id = str(getattr(ctx.event, "id", ""))
    trace_context = (
        cast(Any, {"trace_id": hashlib.md5(event_id.encode("utf-8")).hexdigest()})
        if event_id
        else None
    )
    return langfuse.start_span(
        trace_context=trace_context,
        name="rag.query",
        input={"question": question, "top_k": top_k},
        metadata={
            "event_name": "rag/query",
            "event_id": event_id,
            "user_id": str(ctx.event.data.get("user_id", "")),
            "session_id": str(ctx.event.data.get("session_id", "")),
        },
    )
