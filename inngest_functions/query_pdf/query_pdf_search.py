from rag_types import RAGSearchResult
from services import BM25SparseEncoder, QdrantVectorStore, embed_texts


def search_contexts(question: str, top_k: int = 5) -> RAGSearchResult:
    """Run hybrid retrieval (dense + BM25 sparse) and return top contexts."""
    dense_query_vec = embed_texts([question])[0]
    sparse_encoder = BM25SparseEncoder()
    # Fit on the query to obtain BM25-style sparse query weights.
    sparse_encoder.fit([question])
    sparse_query_vec = sparse_encoder.encode_query(question)

    store = QdrantVectorStore()
    found = store.search(
        dense_query_vec,
        top_k=top_k,
        sparse_query_vector=sparse_query_vec,
        prefetch_k=max(top_k * 3, top_k),
    )
    return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
