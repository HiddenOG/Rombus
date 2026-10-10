"""Before going public: private notebooks, usage limits, upload checks, custom code switch, cleanup."""

import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from app.access import reset_limits
from app.config import settings
from app.session import DataSession, session_manager

ALICE = {"X-Visitor-Id": "alice-visitor-000000001"}
BOB = {"X-Visitor-Id": "bob-visitor-00000000001"}
CSV = b"x,label\n" + b"\n".join(f"{i},{i % 2}".encode() for i in range(30))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    reset_limits()

    async def fake_copilot(session, message):
        return {"response": "ok", "suggested_actions": []}

    monkeypatch.setattr("app.routers.chat.get_copilot_response", fake_copilot)
    with TestClient(app) as c:
        yield c


def upload(client, headers, csv=CSV, name="d.csv"):
    return client.post("/api/upload", files={"file": (name, csv, "text/csv")}, headers=headers)


class TestPrivateNotebooks:
    def test_visitor_id_required(self, client):
        assert upload(client, {}).status_code == 401
        assert client.get("/api/sessions").status_code == 401
        assert upload(client, {"X-Visitor-Id": "short"}).status_code == 401

    def test_others_cannot_see_open_change_or_delete(self, client):
        sid = upload(client, ALICE).json()["session_id"]
        assert [s["session_id"] for s in client.get("/api/sessions", headers=ALICE).json()["sessions"]] == [sid]
        assert client.get("/api/sessions", headers=BOB).json()["sessions"] == []

        assert client.get(f"/api/session/{sid}", headers=BOB).status_code == 404
        assert client.get(f"/api/session/{sid}/cells", headers=BOB).status_code == 404
        action = {"session_id": sid, "tool": "select_target", "arguments": {"column": "label"}}
        assert client.post("/api/action", json=action, headers=BOB).status_code == 404
        assert client.post("/api/chat", json={"session_id": sid, "message": "hi"}, headers=BOB).status_code == 404
        assert client.delete(f"/api/session/{sid}", headers=BOB).status_code == 404

        assert client.get(f"/api/session/{sid}", headers=ALICE).status_code == 200  # still there for Alice

    def test_old_unowned_notebook_is_claimed_by_first_opener(self, client):
        legacy = session_manager.create_session(pd.DataFrame({"x": [1, 2]}), "old.csv", owner=None)
        assert client.get(f"/api/session/{legacy.session_id}", headers=ALICE).status_code == 200
        assert client.get(f"/api/session/{legacy.session_id}", headers=BOB).status_code == 404


class TestLimits:
    def test_ai_messages_per_minute(self, client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMITS", True)
        monkeypatch.setattr(settings, "AI_PER_MINUTE", 3)
        sid = upload(client, ALICE).json()["session_id"]
        codes = [client.post("/api/chat", json={"session_id": sid, "message": "hi"}, headers=ALICE).status_code
                 for _ in range(4)]
        assert codes == [200, 200, 200, 429]

    def test_daily_ai_total(self, client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMITS", True)
        monkeypatch.setattr(settings, "AI_DAILY_LIMIT", 2)
        sid_a = upload(client, ALICE).json()["session_id"]
        sid_b = upload(client, BOB).json()["session_id"]
        assert client.post("/api/chat", json={"session_id": sid_a, "message": "hi"}, headers=ALICE).status_code == 200
        assert client.post("/api/chat", json={"session_id": sid_b, "message": "hi"}, headers=BOB).status_code == 200
        blocked = client.post("/api/chat", json={"session_id": sid_a, "message": "hi"}, headers=ALICE)
        assert blocked.status_code == 429 and "Add step" in blocked.json()["detail"]

    def test_uploads_per_hour(self, client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMITS", True)
        monkeypatch.setattr(settings, "UPLOADS_PER_HOUR", 2)
        assert [upload(client, ALICE).status_code for _ in range(3)] == [200, 200, 429]


class TestUploadChecks:
    def test_wrong_file_type(self, client):
        assert upload(client, ALICE, name="evil.exe").status_code == 400

    def test_too_many_rows(self, client, monkeypatch):
        monkeypatch.setattr(settings, "MAX_ROWS", 10)
        response = upload(client, ALICE)
        assert response.status_code == 413 and "too big" in response.json()["detail"]


class TestCustomCodeSwitch:
    def test_off_hides_and_refuses(self, client, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", False)
        assert client.get("/api/config").json()["custom_code"] is False
        names = {t["name"] for t in client.get("/api/tools").json()["tools"]}
        assert not names & {"custom_code", "custom_view", "custom_models"}

        sid = upload(client, ALICE).json()["session_id"]
        client.post("/api/action", json={"session_id": sid, "tool": "select_target", "arguments": {"column": "label"}},
                    headers=ALICE)
        info = client.post(f"/api/session/{sid}/code", json={"tool": "plot_histogram", "args": {"column": "x"}},
                           headers=ALICE).json()
        assert info["custom_tool"] is None and "switched off" in info["protected"]
        sneaky = {"session_id": sid, "tool": "custom_view", "arguments": {"code": "result = 1"}}
        assert client.post("/api/action", json=sneaky, headers=ALICE).json()["success"] is False


class TestCleanup:
    def test_old_notebooks_deleted_only_when_enabled(self, client, monkeypatch):
        old = session_manager.create_session(pd.DataFrame({"x": [1]}), "old.csv", owner="someone-000000000001")
        old.updated_at = time.time() - 10 * 3600
        session_manager._db.save({**session_manager._db.get(old.session_id), "updated_at": old.updated_at})

        assert session_manager.cleanup_expired() == 0  # TTL 0 = keep forever (local default)
        monkeypatch.setattr(settings, "SESSION_TTL_HOURS", 5)
        assert session_manager.cleanup_expired() == 1
        assert session_manager.get_session(old.session_id) is None
