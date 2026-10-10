"""Accounts: signed-in users own notebooks across devices; visitors' notebooks move in on sign-in.

Tokens here are signed with a throwaway key made by the test, standing in for Supabase's,
so nothing talks to the real Supabase.
"""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from app import access, app
from app.config import settings
from app.session import session_manager

PROJECT = "https://test-project.supabase.co"
KEY = ec.generate_private_key(ec.SECP256R1())
VISITOR = "visitor-before-login-001"
CSV = b"x,y\n" + b"\n".join(f"{i},{i % 2}".encode() for i in range(20))


def token(user_id="user-123", key=KEY, expires_in=3600, issuer=f"{PROJECT}/auth/v1"):
    claims = {"sub": user_id, "aud": "authenticated", "iss": issuer, "exp": int(time.time()) + expires_in}
    return jwt.encode(claims, key, algorithm="ES256")


def headers(visitor=VISITOR, **token_args):
    """A signed-in browser: its visitor id plus a login token."""
    return {"X-Visitor-Id": visitor, "Authorization": f"Bearer {token(**token_args)}"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "SUPABASE_URL", PROJECT)
    monkeypatch.setattr(access, "_signing_key", lambda _token: KEY.public_key())
    with TestClient(app) as c:
        yield c


def upload(client, h):
    return client.post("/api/upload", files={"file": ("d.csv", CSV, "text/csv")}, headers=h).json()["session_id"]


def ids(client, h):
    return {n["session_id"] for n in client.get("/api/sessions", headers=h).json()["sessions"]}


def test_account_notebooks_follow_the_user_to_another_device(client):
    sid = upload(client, headers(visitor="laptop-browser-00000001"))
    phone = headers(visitor="phone-browser-000000001")
    assert sid in ids(client, phone)
    assert client.get(f"/api/session/{sid}", headers=phone).status_code == 200
    # Someone else, and the same browser signed out, see nothing
    assert sid not in ids(client, headers(user_id="someone-else"))
    assert client.get(f"/api/session/{sid}", headers={"X-Visitor-Id": "laptop-browser-00000001"}).status_code == 404


def test_signing_in_moves_the_visitors_notebooks_into_the_account(client):
    anonymous = {"X-Visitor-Id": VISITOR}
    sid = upload(client, anonymous)
    assert sid not in ids(client, headers())

    r = client.post("/api/account/claim", headers=headers())
    assert r.status_code == 200 and r.json()["moved"] == 1
    assert sid in ids(client, headers())
    assert client.get(f"/api/session/{sid}", headers=anonymous).status_code == 404


def test_claiming_needs_a_login(client):
    assert client.post("/api/account/claim", headers={"X-Visitor-Id": VISITOR}).status_code == 401


@pytest.mark.parametrize("bad", [
    {"expires_in": -60},                                    # expired
    {"key": ec.generate_private_key(ec.SECP256R1())},       # signed by someone else
    {"issuer": "https://evil.supabase.co/auth/v1"},         # another project's token
])
def test_bad_tokens_are_refused_not_treated_as_anonymous(client, bad):
    r = client.get("/api/sessions", headers=headers(**bad))
    assert r.status_code == 401


def test_without_supabase_tokens_are_refused(client, monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", "")
    assert client.get("/api/sessions", headers=headers()).status_code == 401
    assert client.get("/api/sessions", headers={"X-Visitor-Id": VISITOR}).status_code == 200


def test_cleanup_keeps_account_notebooks(client, monkeypatch):
    mine = upload(client, headers())
    anon = upload(client, {"X-Visitor-Id": "short-lived-visitor-001"})
    for sid in (mine, anon):
        s = session_manager.get_session(sid)
        session_manager.save(s)
        s.updated_at = 0
        session_manager._db.save({**session_manager._db.get(sid), "updated_at": 0})
    monkeypatch.setattr(settings, "SESSION_TTL_HOURS", 1)
    session_manager.cleanup_expired()
    assert session_manager.get_session(mine) is not None
    assert session_manager.get_session(anon) is None
