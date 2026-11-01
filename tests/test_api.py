import pytest
from fastapi.testclient import TestClient

from lawrag import api

from .conftest import FakeLLM


@pytest.fixture
def client(index, monkeypatch):
    monkeypatch.setattr(api.HybridIndex, "load", classmethod(lambda cls, _dir: index))
    monkeypatch.setattr(api, "get_llm", lambda: FakeLLM("Article 33 requires production within 24 hours [1]."))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["chunks"] == 3 and body["llm"] == "fake/fake-1"


def test_ask_returns_answer_with_citations(client):
    resp = client.post("/api/ask", json={"question": "How soon after arrest must I see a magistrate?", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"][0]["n"] == 1
    assert len(body["sources"]) == 2


@pytest.mark.parametrize("payload", [{"question": "hi"}, {"question": "valid question", "mode": "magic"},
                                     {"question": "valid question", "k": 50}])
def test_ask_validates_input(client, payload):
    assert client.post("/api/ask", json=payload).status_code == 422


def test_llm_failure_is_a_clean_502(client, monkeypatch):
    def boom(*_):
        raise RuntimeError("provider down")

    monkeypatch.setattr(api.state["assistant"].llm, "generate", boom)
    resp = client.post("/api/ask", json={"question": "Are citizens equal?"})
    assert resp.status_code == 502 and "provider down" in resp.json()["detail"]


def test_search_endpoint(client):
    hits = client.get("/api/search", params={"q": "equal before law", "k": 1}).json()
    assert hits[0]["id"] == "constitution-en:27#0"


def test_home_serves_ui(client):
    assert "Bangladesh Law Assistant" in client.get("/").text
