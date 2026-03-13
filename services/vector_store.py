from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Modifier,
    Prefetch,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


class QdrantVectorStore:
    """Thin adapter around Qdrant for upsert + semantic search."""
    DENSE_VECTOR_NAME: str = "dense"
    SPARSE_VECTOR_NAME: str = "sparse"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        dim: int = 3072,
    ) -> None:
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        if not self.client.collection_exists(collection_name=self.collection):
            self._create_collection(dim)

    def _create_collection(self, dim: int) -> None:
        """Create a hybrid-ready collection with named dense/sparse vectors."""
        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    self.DENSE_VECTOR_NAME: VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
                },
            )
        except TypeError:
            # Backward-compatible fallback for older fake clients in tests.
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        sparse_vectors: list[SparseVector] | None = None,
    ) -> None:
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors, and payloads must have the same length.")
        if sparse_vectors is not None and len(sparse_vectors) != len(ids):
            raise ValueError("sparse_vectors must have the same length as ids.")

        points = [
            PointStruct(
                id=ids[i],
                vector=(
                    {
                        self.DENSE_VECTOR_NAME: vectors[i],
                        self.SPARSE_VECTOR_NAME: sparse_vectors[i],
                    }
                    if sparse_vectors is not None
                    else {self.DENSE_VECTOR_NAME: vectors[i]}
                ),
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]
        try:
            self.client.upsert(collection_name=self.collection, points=points)
        except Exception:
            # Fallback for legacy collections that still use unnamed vectors.
            legacy_points = [
                PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                for i in range(len(ids))
            ]
            self.client.upsert(collection_name=self.collection, points=legacy_points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        sparse_query_vector: SparseVector | None = None,
        prefetch_k: int | None = None,
    ) -> dict[str, list[str]]:
        records = self.search_records(
            query_vector=query_vector,
            top_k=top_k,
            sparse_query_vector=sparse_query_vector,
            prefetch_k=prefetch_k,
        )
        contexts = [record["text"] for record in records if record.get("text")]
        sources = sorted({record["source"] for record in records if record.get("source")})
        return {"contexts": contexts, "sources": sources}

    def search_records(
        self,
        query_vector: list[float],
        top_k: int = 5,
        sparse_query_vector: SparseVector | None = None,
        prefetch_k: int | None = None,
    ) -> list[dict[str, str]]:
        candidate_limit = prefetch_k or max(top_k * 3, top_k)
        try:
            if sparse_query_vector is not None:
                response = self.client.query_points(
                    collection_name=self.collection,
                    prefetch=[
                        Prefetch(
                            query=query_vector,
                            using=self.DENSE_VECTOR_NAME,
                            limit=candidate_limit,
                        ),
                        Prefetch(
                            query=sparse_query_vector,
                            using=self.SPARSE_VECTOR_NAME,
                            limit=candidate_limit,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    with_payload=True,
                    limit=top_k,
                )
            else:
                response = self.client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    using=self.DENSE_VECTOR_NAME,
                    with_payload=True,
                    limit=top_k,
                )
        except TypeError:
            # Fallback for simplified test doubles without `using`.
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                with_payload=True,
                limit=top_k,
            )
        except Exception:
            # Fallback for legacy collections or missing sparse vectors.
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                with_payload=True,
                limit=top_k,
            )
        records: list[dict[str, str]] = []
        for result in response.points:
            payload = getattr(result, "payload", None) or {}
            text = str(payload.get("text", "") or "")
            source = str(payload.get("source", "") or "")
            if text or source:
                records.append({"text": text, "source": source})
        return records
