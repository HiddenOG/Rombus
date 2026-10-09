"""Tests for the local Ollama copilot (fake Ollama server, no real model needed)."""

import asyncio
import json

import httpx
import pandas as pd
import pytest

from app import ollama_client
from app.config import settings
from app.copilot_prompt import STATIC_INSTRUCTIONS
from app.session import DataSession
from app.tools import BEFORE_SPLIT_TOOLS


@pytest.fixture
def session():
    df = pd.DataFrame({"age": [20, 30, None, 40], "churned": ["yes", "no", "no", "yes"]})
    s = DataSession(df, "people.csv")
    s.memory = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    return s


def fake_ollama(monkeypatch, handler):
    """Route the client's HTTP calls to `handler(request) -> httpx.Response`."""
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        ollama_client.httpx, "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )


def reply(message):
    return lambda request: httpx.Response(200, json={"message": message})


def ask(session, text="what now?"):
    return asyncio.run(ollama_client.get_ollama_response(session, text))


class TestRequest:
    def test_layout_keeps_static_part_first(self, monkeypatch, session):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "ok"}})

        fake_ollama(monkeypatch, handler)
        ask(session, "what now?")

        messages = seen["messages"]
        assert messages[0] == {"role": "system", "content": STATIC_INSTRUCTIONS}
        assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]
        assert messages[-1]["content"].startswith("SESSION STATE")
        assert messages[-1]["content"].endswith("USER MESSAGE: what now?")
        assert seen["think"] is False
        assert seen["model"] == settings.OLLAMA_MODEL

    def test_only_allowed_tools_offered_before_lock(self, monkeypatch, session):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "ok"}})

        fake_ollama(monkeypatch, handler)
        ask(session)
        assert {t["function"]["name"] for t in seen["tools"]} == BEFORE_SPLIT_TOOLS


class TestParsing:
    def test_text_and_tool_call(self, monkeypatch, session):
        fake_ollama(monkeypatch, reply({
            "content": "Let's pick the target.",
            "tool_calls": [{"function": {"name": "select_target",
                                         "arguments": {"column": "churned", "explanation": "it's the outcome"}}}],
        }))
        out = ask(session)
        assert out["response"] == "Let's pick the target."
        assert out["suggested_actions"] == [
            {"tool": "select_target", "arguments": {"column": "churned"}, "explanation": "it's the outcome"}
        ]

    def test_arguments_as_json_string(self, monkeypatch, session):
        fake_ollama(monkeypatch, reply({
            "content": "",
            "tool_calls": [{"function": {"name": "select_target", "arguments": '{"column": "churned"}'}}],
        }))
        out = ask(session)
        assert out["suggested_actions"][0]["arguments"] == {"column": "churned"}
        assert out["response"]  # fallback sentence when the model only made calls

    def test_unknown_tool_dropped(self, monkeypatch, session):
        fake_ollama(monkeypatch, reply({"content": "x", "tool_calls": [{"function": {"name": "rm_rf", "arguments": {}}}]}))
        assert ask(session)["suggested_actions"] == []

    def test_thinking_hidden(self, monkeypatch, session):
        fake_ollama(monkeypatch, reply({"content": "<think>hmm let me see</think>Pick a target."}))
        assert ask(session)["response"] == "Pick a target."


class TestFailures:
    def test_model_not_downloaded(self, monkeypatch, session):
        fake_ollama(monkeypatch, lambda r: httpx.Response(404, json={"error": "model not found"}))
        assert f"ollama pull {settings.OLLAMA_MODEL}" in ask(session)["response"]

    def test_ollama_not_running(self, monkeypatch, session):
        def handler(request):
            raise httpx.ConnectError("refused")

        fake_ollama(monkeypatch, handler)
        assert ask(session)["response"] == ollama_client.NOT_RUNNING_MESSAGE

    def test_too_slow(self, monkeypatch, session):
        def handler(request):
            raise httpx.ReadTimeout("slow")

        fake_ollama(monkeypatch, handler)
        assert ask(session)["response"] == ollama_client.SLOW_MESSAGE


def test_provider_switch(monkeypatch, session):
    from app import copilot

    async def fake_local(s, m):
        return {"response": "local", "suggested_actions": []}

    async def fake_cloud(s, m):
        return {"response": "cloud", "suggested_actions": []}

    monkeypatch.setattr("app.ollama_client.get_ollama_response", fake_local)
    monkeypatch.setattr("app.gemini.get_gemini_response", fake_cloud)

    monkeypatch.setattr(settings, "AI_PROVIDER", "ollama")
    assert asyncio.run(copilot.get_copilot_response(session, "x"))["response"] == "local"
    monkeypatch.setattr(settings, "AI_PROVIDER", "gemini")
    assert asyncio.run(copilot.get_copilot_response(session, "x"))["response"] == "cloud"
