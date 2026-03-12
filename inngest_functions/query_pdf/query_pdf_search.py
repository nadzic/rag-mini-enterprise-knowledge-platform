from rag_types import RAGSearchResult
from services import (
    BM25SparseEncoder,
    QdrantVectorStore,
    embed_texts,
    rerank_enabled,
    rerank_records,
    rerank_top_n,
)


def search_contexts(question: str, top_k: int = 5) -> RAGSearchResult:
    """Run hybrid retrieval (dense + BM25 sparse) and return top contexts."""
    dense_query_vec = embed_texts([question])[0]
    sparse_encoder = BM25SparseEncoder()
    # Fit on the query to obtain BM25-style sparse query weights.
    sparse_encoder.fit([question])
    sparse_query_vec = sparse_encoder.encode_query(question)

    first_stage_k = max(top_k, rerank_top_n()) if rerank_enabled() else top_k

    store = QdrantVectorStore()
    records = store.search_records(
        dense_query_vec,
        top_k=first_stage_k,
        sparse_query_vector=sparse_query_vec,
        prefetch_k=max(first_stage_k * 3, first_stage_k),
    )

    text_records = [record for record in records if record.get("text")]
    selected_records = rerank_records(question, text_records, top_k=top_k)
    contexts = [record["text"] for record in selected_records if record.get("text")]
    sources = sorted({record["source"] for record in selected_records if record.get("source")})
    return RAGSearchResult(contexts=contexts, sources=sources)
