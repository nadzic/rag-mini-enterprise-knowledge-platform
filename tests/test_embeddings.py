from types import SimpleNamespace

import services.embeddings as embeddings


class _FakeEmbeddingsAPI:
    def create(self, model, input):
        assert model == embeddings.EMBED_MODEL
        assert input == ["hello", "world"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2]),
                SimpleNamespace(embedding=[0.3, 0.4]),
            ]
        )


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddingsAPI()


def test_embed_texts_returns_embedding_vectors(monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "_get_openai_client", lambda: _FakeClient())
    vectors = embeddings.embed_texts(["hello", "world"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
