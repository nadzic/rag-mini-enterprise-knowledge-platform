import inngest

from rag_types import RAGChunkAndSource, RAGUpsertResult
from services.langfuse_client import flush_langfuse

from .client import inngest_client
from .ingest_pdf_finalize_failure import finalize_failure
from .ingest_pdf_load import load_chunks_and_source
from .ingest_pdf_start_span import start_root_span
from .ingest_pdf_upsert import upsert_chunks


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    """Orchestrate PDF loading/chunking, embedding/upsert, and tracing output."""
    root_span = start_root_span(ctx)

    load_span = root_span.start_span(name="load-and-chunk-pdf") if root_span else None
    try:
        async def _load_step() -> RAGChunkAndSource:
            return load_chunks_and_source(ctx)

        chunks_and_source = await ctx.step.run(
            "load-and-chunk-pdf",
            _load_step,
            output_type=RAGChunkAndSource,
        )
        if load_span is not None:
            load_span.update(
                output={
                    "source_id": chunks_and_source.source_id,
                    "num_chunks": len(chunks_and_source.chunks),
                }
            )
    except Exception as exc:
        finalize_failure(root_span, load_span, exc)
        raise
    else:
        if load_span is not None:
            load_span.end()

    upsert_span = root_span.start_span(name="embed-and-upsert") if root_span else None
    try:
        async def _upsert_step() -> RAGUpsertResult:
            return upsert_chunks(chunks_and_source)

        ingested = await ctx.step.run(
            "embed-and-upsert",
            _upsert_step,
            output_type=RAGUpsertResult,
        )
        if upsert_span is not None:
            upsert_span.update(output={"ingested": ingested.ingested})
    except Exception as exc:
        finalize_failure(root_span, upsert_span, exc)
        raise
    else:
        if upsert_span is not None:
            upsert_span.end()

    result = ingested.model_dump()
    if root_span is not None:
        root_span.update(output=result)
        root_span.end()
    flush_langfuse()
    return result
