import sys
from pathlib import Path

from fastapi.testclient import TestClient


API_SERVICE_SRC = Path(__file__).resolve().parents[1] / "api_service" / "src" / "api"
if str(API_SERVICE_SRC) not in sys.path:
    sys.path.insert(0, str(API_SERVICE_SRC))

import main as api_main  # noqa: E402
from api.routes import rag as rag_route  # noqa: E402


client = TestClient(api_main.app)


def test_api_root_returns_hello_world() -> None:
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_health_endpoint_returns_service_metadata() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "api_service",
        "version": "0.1.0",
    }


def test_rag_query_returns_expected_response(monkeypatch) -> None:
    async def _fake_run_rag_query(question: str, top_k: int):
        assert question == "What is this platform?"
        assert top_k == 3
        return {
            "answer": "It is a document-grounded Q&A backend.",
            "sources": ["architecture.pdf", "onboarding.pdf"],
            "num_contexts": 3,
        }

    monkeypatch.setattr(rag_route, "run_rag_query", _fake_run_rag_query)

    response = client.post(
        "/api/rag/query",
        json={"question": "What is this platform?", "top_k": 3},
    )
    assert response.status_code == 200
    assert response.json() == {
        "answer": "It is a document-grounded Q&A backend.",
        "sources": ["architecture.pdf", "onboarding.pdf"],
        "num_contexts": 3,
    }


def test_rag_query_uses_default_top_k(monkeypatch) -> None:
    seen: dict[str, int | str] = {}

    async def _fake_run_rag_query(question: str, top_k: int):
        seen["question"] = question
        seen["top_k"] = top_k
        return {"answer": "ok", "sources": [], "num_contexts": top_k}

    monkeypatch.setattr(rag_route, "run_rag_query", _fake_run_rag_query)

    response = client.post(
        "/api/rag/query",
        json={"question": "Use default top_k"},
    )
    assert response.status_code == 200
    assert seen == {"question": "Use default top_k", "top_k": 5}
    assert response.json() == {"answer": "ok", "sources": [], "num_contexts": 5}


def test_rag_query_rejects_invalid_payload() -> None:
    response = client.post(
        "/api/rag/query",
        json={"question": "", "top_k": 0},
    )
    assert response.status_code == 422
