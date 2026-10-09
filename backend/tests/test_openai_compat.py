"""Tests for the OpenAI-compatible inference server connector (fake server, no network)."""

import asyncio
import json

import httpx
import pandas as pd
import pytest

from app import openai_compat_client as client_mod
from app.config import settings
from app.session import DataSession


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(settings, "AI_BASE_URL", "http://fake-server/v1")
    monkeypatch.setattr(settings, "AI_MODEL", "qwen-test")
    monkeypatch.setattr(settings, "AI_API_KEY", "secret")
    return DataSession(pd.DataFrame({"age": [1, 2], "churned": ["y", "n"]}), "d.csv")


def fake_server(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(client_mod.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def ask(session):
    return asyncio.run(client_mod.get_openai_compatible_response(session, "what now?"))


def test_request_and_tool_call(monkeypatch, session):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "Pick the target.",
            "tool_calls": [{"type": "function", "function": {
                "name": "select_target", "arguments": '{"column": "churned", "explanation": "outcome"}'}}],
        }}]})

    fake_server(monkeypatch, handler)
    out = ask(session)
    assert seen["url"] == "http://fake-server/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "qwen-test"
    assert out["response"] == "Pick the target."
    assert out["suggested_actions"] == [{"tool": "select_target", "arguments": {"column": "churned"}, "explanation": "outcome"}]


@pytest.mark.parametrize("status,expected", [
    (401, client_mod.KEY_MESSAGE),
    (429, client_mod.QUOTA_MESSAGE),
    (503, client_mod.BUSY_MESSAGE),
    (400, client_mod.GENERIC_MESSAGE),
])
def test_error_statuses(monkeypatch, session, status, expected):
    fake_server(monkeypatch, lambda r: httpx.Response(status, json={"error": "x"}))
    assert ask(session)["response"] == expected


def test_unreachable(monkeypatch, session):
    def handler(request):
        raise httpx.ConnectError("refused")

    fake_server(monkeypatch, handler)
    assert ask(session)["response"] == client_mod.NOT_REACHABLE_MESSAGE


def test_not_configured(monkeypatch, session):
    monkeypatch.setattr(settings, "AI_BASE_URL", "")
    assert "Set AI_BASE_URL" in ask(session)["response"]
