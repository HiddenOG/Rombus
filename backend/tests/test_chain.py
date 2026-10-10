"""The provider fallback chain: try in order, skip recently failed providers, say who answered."""

import asyncio

import pandas as pd
import pytest

from app import copilot
from app.config import settings
from app.copilot_prompt import failure
from app.session import DataSession


@pytest.fixture
def session():
    return DataSession(pd.DataFrame({"x": [1, 2], "y": [0, 1]}), "d.csv")


@pytest.fixture
def providers(monkeypatch):
    """Fake providers: each name maps to a failure reason, or None to answer."""
    behaviour = {}
    calls = []

    def make(name):
        async def ask(session, message):
            calls.append(name)
            reason = behaviour.get(name)
            return failure(f"{name} failed", reason) if reason else {"response": f"hi from {name}", "suggested_actions": []}
        return ask

    names = ["groq", "gemini", "ollama"]
    monkeypatch.setattr(copilot, "_providers", lambda: {n: (n.title(), make(n), lambda n=n: f"{n}-model") for n in names})
    monkeypatch.setattr(copilot, "_skip_until", {})
    monkeypatch.setattr(settings, "AI_CHAIN", "groq,gemini,ollama")
    return behaviour, calls


def ask(session):
    return asyncio.run(copilot.get_copilot_response(session, "hello"))


def test_first_provider_answers(session, providers):
    _, calls = providers
    out = ask(session)
    assert out["response"] == "hi from groq" and out["provider"] == "Groq" and out["model"] == "groq-model"
    assert calls == ["groq"]
    assert "fallback_note" not in out


def test_falls_back_and_says_why(session, providers):
    behaviour, calls = providers
    behaviour["groq"] = "quota"
    out = ask(session)
    assert out["provider"] == "Gemini"
    assert "Groq was unavailable (quota)" in out["fallback_note"]
    assert calls == ["groq", "gemini"]


def test_failed_provider_is_skipped_for_a_while(session, providers):
    behaviour, calls = providers
    behaviour["groq"] = "quota"
    ask(session)
    behaviour["groq"] = None  # even if it recovers, it's in its cool-down
    ask(session)
    assert calls == ["groq", "gemini", "gemini"]


def test_all_fail(session, providers):
    behaviour, _ = providers
    behaviour.update(groq="quota", gemini="busy", ollama="unreachable")
    out = ask(session)
    assert out["failure"] == "unreachable"
    assert "None of the AI providers could answer" in out["response"]


def test_last_provider_always_tried(session, providers):
    behaviour, calls = providers
    behaviour.update(groq="quota", gemini="quota", ollama="timeout")
    ask(session)
    behaviour.clear()
    out = ask(session)  # everything is cooling down, but the last one is still tried
    assert out["provider"] == "Ollama"


def test_single_provider_mode_without_chain(session, providers, monkeypatch):
    monkeypatch.setattr(settings, "AI_CHAIN", "")
    monkeypatch.setattr(settings, "AI_PROVIDER", "gemini")
    assert ask(session)["provider"] == "Gemini"


def test_unknown_provider_name(session, providers, monkeypatch):
    monkeypatch.setattr(settings, "AI_CHAIN", "groq,chatgpt")
    assert ask(session)["failure"] == "config"
