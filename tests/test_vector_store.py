from types import SimpleNamespace

import pytest

import services.vector_store as vector_store


class FakeQdrantClient:
    def __init__(self, url, timeout):
        self.url = url
        self.timeout = timeout
        self.exists = False
        self.created = None
        self.upsert_calls = []
        self.query_result = []

    def collection_exists(self, collection_name):
        return self.exists

    def create_collection(self, collection_name, vectors_config):
        self.created = (collection_name, vectors_config)

    def upsert(self, collection_name, points):
        self.upsert_calls.append((collection_name, points))

    def query_points(self, collection_name, query, with_payload, limit):
        return SimpleNamespace(points=self.query_result)


def test_init_creates_collection_when_missing(monkeypatch) -> None:
    fake_client = FakeQdrantClient(url="http://unused", timeout=30)
    monkeypatch.setattr(vector_store, "QdrantClient", lambda url, timeout: fake_client)
    store = vector_store.QdrantVectorStore(collection="my_docs", dim=4)
    assert store.collection == "my_docs"
    assert fake_client.created is not None
    assert fake_client.created[0] == "my_docs"
    assert fake_client.created[1].size == 4


def test_upsert_raises_on_length_mismatch(monkeypatch) -> None:
    fake_client = FakeQdrantClient(url="http://unused", timeout=30)
    fake_client.exists = True
    monkeypatch.setattr(vector_store, "QdrantClient", lambda url, timeout: fake_client)
    store = vector_store.QdrantVectorStore()
    with pytest.raises(ValueError):
        store.upsert(ids=["1"], vectors=[[1.0], [2.0]], payloads=[{"text": "a"}])


def test_search_returns_contexts_and_unique_sorted_sources(monkeypatch) -> None:
    fake_client = FakeQdrantClient(url="http://unused", timeout=30)
    fake_client.exists = True
    fake_client.query_result = [
        SimpleNamespace(payload={"text": "ctx 1", "source": "b.pdf"}),
        SimpleNamespace(payload={"text": "ctx 2", "source": "a.pdf"}),
        SimpleNamespace(payload={"text": "", "source": "a.pdf"}),
        SimpleNamespace(payload={"source": "c.pdf"}),
    ]
    monkeypatch.setattr(vector_store, "QdrantClient", lambda url, timeout: fake_client)
    store = vector_store.QdrantVectorStore()
    result = store.search([0.1, 0.2], top_k=4)
    assert result == {"contexts": ["ctx 1", "ctx 2"], "sources": ["a.pdf", "b.pdf", "c.pdf"]}
