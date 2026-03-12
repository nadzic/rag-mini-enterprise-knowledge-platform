import uuid

from rag_types import RAGChunkAndSource, RAGUpsertResult
from services import QdrantVectorStore, build_bm25_encoder_and_sparse_chunks, embed_texts


def upsert_chunks(chunks_and_source: RAGChunkAndSource) -> RAGUpsertResult:
    """Embed PDF chunks and upsert deterministic vectors into Qdrant."""
    chunks = chunks_and_source.chunks
    source_id = chunks_and_source.source_id
    dense_vectors = embed_texts(chunks)
    _, sparse_vectors = build_bm25_encoder_and_sparse_chunks(chunks)
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{index}"))
        for index in range(len(chunks))
    ]
    payloads = [{"source": source_id, "text": chunks[index]} for index in range(len(chunks))]
    store = QdrantVectorStore()
    store.upsert(ids, dense_vectors, payloads, sparse_vectors=sparse_vectors)
    return RAGUpsertResult(ingested=len(chunks))
