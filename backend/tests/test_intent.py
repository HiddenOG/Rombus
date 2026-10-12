"""The safety net: obvious requests get a Run button even when the AI answers in words only."""

import asyncio

import pandas as pd
import pytest

from app import copilot
from app.intent import obvious_step
from app.session import DataSession
from app.tools import execute_tool


@pytest.fixture
def session():
    df = pd.DataFrame({
        "society": ["A", "B", None, "A"] * 10, "balcony": [1.0, None, 2.0, 1.0] * 10,
        "area_type": ["x", "y", "x", "x"] * 10, "price": [10, 20, 30, 10] * 10,
    })
    s = DataSession(df, "houses.csv")
    assert execute_tool("select_target", s, {"column": "price"})["success"]
    return s


def step(session, message):
    found = obvious_step(session, message)
    return found and (found["tool"], found["arguments"])


@pytest.mark.parametrize("message, tool", [
    ("drop duplicates", "remove_duplicates"),
    ("DROP duplicates", "remove_duplicates"),
    ("have you droped duplocates i mean", "remove_duplicates"),   # typos, from a real session
    ("remove the duplicate rows please", "remove_duplicates"),
    ("dedupe", "remove_duplicates"),
    ("fill the missing values", "impute_all_missing"),
    ("handle nulls", "impute_all_missing"),
    ("one-hot encode the text columns", "encode_all_categorical"),
    ("normalize everything", "scale_all_features"),
    ("train some models", "train_models"),
    ("tune the hyperparameters", "auto_ml"),
])
def test_clear_commands(session, message, tool):
    assert step(session, message) == (tool, {})


def test_naming_columns_to_remove(session):
    assert step(session, "drop the society and balcony columns") == ("remove_columns", {"columns": ["society", "balcony"]})
    assert step(session, "remove price") is None  # the target is never removed this way


@pytest.mark.parametrize("message", ["how many duplicates are there?", "what is a good model?", "hello"])
def test_questions_and_chat_are_left_to_the_ai(session, message):
    assert step(session, message) is None


def test_drop_em_after_counting_duplicates(session):
    """The real sequence: check duplicates ran, then 'drop em'."""
    session.memory.append({"role": "assistant", "content": "Action Executed: check_duplicates. Result: Found 30 duplicate rows."})
    session.memory.append({"role": "user", "content": "drop em"})
    assert step(session, "drop em") == ("remove_duplicates", {})


def test_yes_refers_to_the_last_suggestion_past_word_only_replies(session):
    session.memory += [
        {"role": "assistant", "content": "Fill the blanks.\n[Suggested as Run buttons: impute_all_missing]"},
        {"role": "user", "content": "hmm"},
        {"role": "assistant", "content": "Let me run it now:"},  # words only
    ]
    assert step(session, "yes do it") == ("impute_all_missing", {})


def test_never_offers_the_final_exam_or_steps_needing_settings(session):
    session.memory.append({"role": "assistant", "content": "Ready.\n[Suggested as Run buttons: final_exam]"})
    assert step(session, "go ahead") is None


def test_nothing_before_the_test_rows_are_locked():
    s = DataSession(pd.DataFrame({"x": [1, 1, 2], "y": [0, 0, 1]}), "d.csv")
    assert obvious_step(s, "drop duplicates") is None


def test_no_duplicate_button_when_there_are_none(session):
    assert execute_tool("remove_duplicates", session, {})["success"]
    assert step(session, "drop duplicates") is None


def test_the_chain_adds_the_button_only_when_the_ai_gave_none(session, monkeypatch):
    replies = {
        "talk": {"response": "Dropping those duplicate rows now.", "suggested_actions": []},
        "suggests": {"response": "Try this.", "suggested_actions": [{"tool": "check_duplicates", "arguments": {}, "explanation": ""}]},
    }

    async def ask(_session, message):
        return dict(replies[current])

    monkeypatch.setattr(copilot, "_providers", lambda: {"groq": ("Groq", ask, lambda: "m")})
    monkeypatch.setattr(copilot.settings, "AI_CHAIN", "groq")

    current = "talk"
    result = asyncio.run(copilot.get_copilot_response(session, "drop duplicates"))
    assert [a["tool"] for a in result["suggested_actions"]] == ["remove_duplicates"]

    current = "suggests"  # the AI's own choice is never overridden
    result = asyncio.run(copilot.get_copilot_response(session, "drop duplicates"))
    assert [a["tool"] for a in result["suggested_actions"]] == ["check_duplicates"]
