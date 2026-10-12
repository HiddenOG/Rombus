"""The copilot announcing an action without suggesting it ("Let me run it now:" and no Run button)."""

import asyncio

import pandas as pd
import pytest

from app import openai_compat_client as client
from app.copilot_prompt import memory_entry, promises_action
from app.openai_compat_client import Server, get_openai_compatible_response
from app.session import DataSession


@pytest.mark.parametrize("text, promise", [
    ("Let me run it now:", True),                         # seen live from Groq
    ("Dropping those 375 duplicate rows now.", True),     # seen live from Groq
    ("I'll drop them.", True),
    ("Good catch, let’s fix that.", True),
    ("Here's what's left:", True),
    ("Training accuracy is 0.8, so you're done.", False),
    ("You have 375 duplicate rows.", False),
    ("", False),
])
def test_spots_promises(text, promise):
    assert promises_action(text) is promise


def test_history_shows_what_was_suggested():
    reply = {"response": "Remove them.", "suggested_actions": [{"tool": "remove_duplicates", "arguments": {}}]}
    assert memory_entry(reply) == "Remove them.\n[Suggested as Run buttons: remove_duplicates]"
    assert memory_entry({"response": "Done.", "suggested_actions": []}) == "Done."


@pytest.fixture
def session():
    s = DataSession(pd.DataFrame({"x": [1, 1, 2, 3] * 10, "y": [0, 0, 1, 1] * 10}), "d.csv")
    s.lock_count = 0
    return s


def fake_server(monkeypatch, replies):
    """Each request gets the next reply; records the tool_choice it was asked with."""
    asked = []

    async def ask(server, payload):
        asked.append(payload["tool_choice"])
        return replies.pop(0)

    monkeypatch.setattr(client, "_ask", ask)
    return asked


def run(session):
    return asyncio.run(get_openai_compatible_response(session, "drop em", Server("Groq", "https://x", "m", "key")))


def reply(text, actions=()):
    return {"response": text, "suggested_actions": list(actions), "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_a_broken_promise_is_retried_with_a_required_call(session, monkeypatch):
    step = {"tool": "remove_duplicates", "arguments": {}, "explanation": "Duplicates."}
    asked = fake_server(monkeypatch, [reply("Let me run it now:"), reply("", [step])])
    result = run(session)
    assert asked == ["auto", "required"]
    assert result["response"] == "Let me run it now:" and result["suggested_actions"] == [step]
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 10}


def test_says_so_when_the_retry_also_brings_nothing(session, monkeypatch):
    fake_server(monkeypatch, [reply("Let me run it now:"), reply("")])
    assert "Add step" in run(session)["response"]


def test_plain_answers_are_not_retried(session, monkeypatch):
    asked = fake_server(monkeypatch, [reply("You have 4 duplicate rows.")])
    run(session)
    assert asked == ["auto"]
