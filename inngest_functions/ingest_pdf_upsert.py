import uuid

from rag_types import RAGChunkAndSource, RAGUpsertResult
from services import QdrantVectorStore, embed_texts


def upsert_chunks(chunks_and_source: RAGChunkAndSource) -> RAGUpsertResult:
    """Embed PDF chunks and upsert deterministic vectors into Qdrant."""
    chunks = chunks_and_source.chunks
    source_id = chunks_and_source.source_id
    vectors = embed_texts(chunks)
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{index}"))
        for index in range(len(chunks))
    ]
    payloads = [{"source": source_id, "text": chunks[index]} for index in range(len(chunks))]
    store = QdrantVectorStore()
    store.upsert(ids, vectors, payloads)
    return RAGUpsertResult(ingested=len(chunks))
