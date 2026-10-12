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


def test_history_keeps_suggestions_as_data_not_text():
    """Written into the text, models copied the pattern instead of making calls (seen live)."""
    reply = {"response": "Remove them.", "suggested_actions": [{"tool": "remove_duplicates", "arguments": {}}]}
    assert memory_entry(reply) == {"role": "assistant", "content": "Remove them.", "suggested": ["remove_duplicates"]}


def test_written_out_suggestions_become_real_ones():
    """Seen live: "Let's drop the duplicate rows. [Suggested as Run buttons: remove_duplicates]" and no call."""
    from app.copilot_prompt import finish
    reply = finish(["Let's drop them. [Suggested as Run buttons: remove_duplicates, final_exam, impute_missing]\n\n"
                    "_(No step came with this answer. You can add it yourself with **Add step**, or ask again.)_"], [])
    assert reply["response"] == "Let's drop them."
    # final_exam is never offered from text; impute_missing needs settings it doesn't have
    assert [a["tool"] for a in reply["suggested_actions"]] == ["remove_duplicates"]


def test_old_copied_text_is_cleaned_from_history():
    from app.copilot_prompt import recent_history
    s = DataSession(pd.DataFrame({"x": [1, 2]}), "d.csv")
    s.memory = [{"role": "assistant", "content": "Undo first: [Suggested as Run buttons: undo]"}]
    assert recent_history(s, "next")[0]["content"] == "Undo first:"


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
    result = run(session)
    # Shown to the person, but kept apart from the words that go into the history
    assert "Add step" in result["notice"] and "Add step" not in result["response"]


def test_plain_answers_are_not_retried(session, monkeypatch):
    asked = fake_server(monkeypatch, [reply("You have 4 duplicate rows.")])
    run(session)
    assert asked == ["auto"]
