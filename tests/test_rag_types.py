import pytest
from pydantic import ValidationError

from rag_types import RAGChunkAndSource, RAGQueryResult, RAGSearchResult, RAGUpsertResult


def test_rag_chunk_and_source_defaults_source_id_to_none() -> None:
    item = RAGChunkAndSource(chunks=["a", "b"])
    assert item.source_id is None
    assert item.chunks == ["a", "b"]


def test_rag_upsert_result_requires_ingested() -> None:
    result = RAGUpsertResult(ingested=3)
    assert result.ingested == 3


def test_rag_search_result_roundtrip() -> None:
    model = RAGSearchResult(contexts=["c1"], sources=["s1"])
    assert model.model_dump() == {"contexts": ["c1"], "sources": ["s1"]}


def test_rag_query_result_validates_num_contexts_type() -> None:
    with pytest.raises(ValidationError):
        RAGQueryResult(answer="ok", sources=["s1"], num_contexts="two")
