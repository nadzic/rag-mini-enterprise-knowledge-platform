import uuid

import inngest

from rag_types import RAGChunkAndSource, RAGUpsertResult
from services import QdrantVectorStore, embed_texts, load_and_chunk_pdf

from .client import inngest_client


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(context: inngest.Context) -> RAGChunkAndSource:
        pdf_path = context.event.data["pdf_path"]
        source_id = context.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSource(chunks=chunks, source_id=source_id)

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

    chunks_and_source = await ctx.step.run(
        "load-and-chunk-pdf",
        lambda: _load(ctx),
        output_type=RAGChunkAndSource,
    )
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_source),
        output_type=RAGUpsertResult,
    )
    return ingested.model_dump()
