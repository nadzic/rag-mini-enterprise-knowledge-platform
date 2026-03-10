from typing import Any

import inngest

from services.langfuse_client import get_langfuse_client


def start_root_span(ctx: inngest.Context) -> Any:
    """Create the root Langfuse span for the ingest event when enabled."""
    langfuse = get_langfuse_client()
    if langfuse is None:
        return None
    return langfuse.start_span(
        name="rag.ingest_pdf",
        input={
            "pdf_path": str(ctx.event.data.get("pdf_path", "")),
            "source_id": str(ctx.event.data.get("source_id", "")),
        },
        metadata={
            "event_name": "rag/ingest_pdf",
            "event_id": str(getattr(ctx.event, "id", "")),
        },
    )
