"""Tests for the OpenAI-compatible inference server connector (fake server, no network)."""

import asyncio
import json

import httpx
import pandas as pd
import pytest

from app import openai_compat_client as client_mod
from app.openai_compat_client import Server
from app.session import DataSession

SERVER = Server("TestServer", "http://fake-server/v1", "qwen-test", "secret")


@pytest.fixture
def session():
    return DataSession(pd.DataFrame({"age": [1, 2], "churned": ["y", "n"]}), "d.csv")


def fake_server(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(client_mod.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def ask(session, server=SERVER):
    return asyncio.run(client_mod.get_openai_compatible_response(session, "what now?", server))


def test_request_tool_call_and_usage(monkeypatch, session):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {
                "content": "Pick the target.",
                "tool_calls": [{"type": "function", "function": {
                    "name": "select_target", "arguments": '{"column": "churned", "explanation": "outcome"}'}}],
            }}],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 50},
        })

    fake_server(monkeypatch, handler)
    out = ask(session)
    assert seen["url"] == "http://fake-server/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "qwen-test"
    assert out["response"] == "Pick the target."
    assert out["suggested_actions"] == [{"tool": "select_target", "arguments": {"column": "churned"}, "explanation": "outcome"}]
    assert out["usage"] == {"input_tokens": 2000, "output_tokens": 50}
    assert "failure" not in out


@pytest.mark.parametrize("status,reason", [(401, "auth"), (429, "quota"), (503, "busy"), (400, "error")])
def test_error_statuses_give_failure_reasons(monkeypatch, session, status, reason):
    fake_server(monkeypatch, lambda r: httpx.Response(status, json={"error": "x"}))
    out = ask(session)
    assert out["failure"] == reason
    assert "TestServer" in out["response"]  # messages name the provider


def test_unreachable(monkeypatch, session):
    def handler(request):
        raise httpx.ConnectError("refused")

    fake_server(monkeypatch, handler)
    assert ask(session)["failure"] == "unreachable"


def test_not_configured(session):
    assert ask(session, Server("X", "", ""))["failure"] == "config"
    assert ask(session, Server("Hosted", "https://api.example.com/v1", "m", ""))["failure"] == "config"  # key needed


def test_groq_key_accepts_either_name(monkeypatch):
    from app.config import Settings
    monkeypatch.setenv("GROQ_KEY", "gsk_from_short_name")
    assert Settings(_env_file=None).GROQ_API_KEY == "gsk_from_short_name"
