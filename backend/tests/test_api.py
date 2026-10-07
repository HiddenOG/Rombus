"""End-to-end API tests through FastAPI's TestClient (Gemini stubbed)."""

import pytest
from fastapi.testclient import TestClient

from app import app
from app.config import settings

CSV = (
    "size,color,price,label\n"
    + "\n".join(
        f"{i},{'red' if i % 2 else 'blue'},{i * 1.5},{'yes' if i % 3 else 'no'}"
        for i in range(1, 41)
    )
).encode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    async def fake_copilot(session, message):
        return {
            "response": "Pick a target.",
            "suggested_actions": [{"tool": "select_target", "arguments": {"column": "label"}, "explanation": "x"}],
        }

    monkeypatch.setattr("app.routers.chat.get_copilot_response", fake_copilot)
    with TestClient(app) as c:
        yield c


def _upload(client):
    resp = client.post("/api/upload", files={"file": ("data.csv", CSV, "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def _action(client, sid, tool, **arguments):
    resp = client.post("/api/action", json={"session_id": sid, "tool": tool, "arguments": arguments})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health(client):
    assert client.get("/api/health").json()["status"] == "healthy"


def test_full_modeling_flow(client):
    sid = _upload(client)
    assert _action(client, sid, "select_target", column="label")["success"]

    split = _action(client, sid, "train_test_split", test_size=0.25)
    assert split["success"]
    assert "color" in split["message"]  # text feature auto-encoded

    trained = _action(client, sid, "train_models", algorithms=["rf", "lr"])
    assert trained["success"], trained["message"]
    assert trained["evidence"]["best_model"] in ("rf", "lr")


def test_chat_returns_suggestions(client):
    sid = _upload(client)
    resp = client.post("/api/chat", json={"session_id": sid, "message": "what now?"})
    assert resp.status_code == 200
    assert resp.json()["suggested_actions"][0]["tool"] == "select_target"


def test_session_recovery(client):
    sid = _upload(client)
    assert client.get(f"/api/session/{sid}").json()["session_id"] == sid
    assert client.get("/api/session/does-not-exist").status_code == 404


def test_unknown_session(client):
    resp = client.post("/api/action", json={"session_id": "nope", "tool": "undo", "arguments": {}})
    assert resp.status_code == 404
