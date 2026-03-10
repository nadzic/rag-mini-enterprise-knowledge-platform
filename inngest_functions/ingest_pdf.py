import uuid

import inngest

from rag_types import RAGChunkAndSource, RAGUpsertResult
from services import QdrantVectorStore, embed_texts, load_and_chunk_pdf
from services.langfuse_client import flush_langfuse, get_langfuse_client

from .client import inngest_client


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(context: inngest.Context) -> RAGChunkAndSource:
        pdf_path = context.event.data["pdf_path"]
        source_id = context.event.data.get("source_id", str(pdf_path))
        chunks = load_and_chunk_pdf(str(pdf_path))
        return RAGChunkAndSource(chunks=chunks, source_id=str(source_id))

    def _upsert(chunks_and_source: RAGChunkAndSource) -> RAGUpsertResult:
        chunks = chunks_and_source.chunks
        source_id = chunks_and_source.source_id
        vectors = embed_texts(chunks)
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        store = QdrantVectorStore()
        store.upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    langfuse = get_langfuse_client()
    root_span = None
    if langfuse is not None:
        root_span = langfuse.start_span(
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

    load_span = root_span.start_span(name="load-and-chunk-pdf") if root_span else None
    try:
        async def _load_step() -> RAGChunkAndSource:
            return _load(ctx)

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
        if load_span is not None:
            load_span.update(level="ERROR", status_message=str(exc))
            load_span.end()
        if root_span is not None:
            root_span.update(level="ERROR", status_message=str(exc))
            root_span.end()
        flush_langfuse()
        raise
    else:
        if load_span is not None:
            load_span.end()

    upsert_span = root_span.start_span(name="embed-and-upsert") if root_span else None
    try:
        async def _upsert_step() -> RAGUpsertResult:
            return _upsert(chunks_and_source)

        ingested = await ctx.step.run(
            "embed-and-upsert",
            _upsert_step,
            output_type=RAGUpsertResult,
        )
        if upsert_span is not None:
            upsert_span.update(output={"ingested": ingested.ingested})
    except Exception as exc:
        if upsert_span is not None:
            upsert_span.update(level="ERROR", status_message=str(exc))
            upsert_span.end()
        if root_span is not None:
            root_span.update(level="ERROR", status_message=str(exc))
            root_span.end()
        flush_langfuse()
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
