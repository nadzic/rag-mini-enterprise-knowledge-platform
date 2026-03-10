import hashlib
from typing import Any, cast

import inngest

from services.langfuse_client import get_langfuse_client


def start_root_span(ctx: inngest.Context) -> Any:
    """Create the root Langfuse span for the ingest event when enabled."""
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
        name="rag.ingest_pdf",
        input={
            "pdf_path": str(ctx.event.data.get("pdf_path", "")),
            "source_id": str(ctx.event.data.get("source_id", "")),
        },
        metadata={
            "event_name": "rag/ingest_pdf",
            "event_id": event_id,
        },
    )
