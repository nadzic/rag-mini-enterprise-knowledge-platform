from rag_types import RAGSearchResult
from services import QdrantVectorStore, embed_texts


def search_contexts(question: str, top_k: int = 5) -> RAGSearchResult:
    """Embed the question and return the top matching contexts from Qdrant."""
    query_vec = embed_texts([question])[0]
    store = QdrantVectorStore()
    found = store.search(query_vec, top_k)
    return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
