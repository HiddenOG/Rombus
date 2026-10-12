"""Judgement-call steps (outliers, dropping rows, new features...) only when the user asks for them."""

import asyncio

import pandas as pd
import pytest

from app import copilot
from app.copilot_prompt import STATIC_INSTRUCTIONS
from app.intent import asked_for
from app.session import DataSession
from app.tools import execute_tool


@pytest.fixture
def session():
    s = DataSession(pd.DataFrame({"sqft": range(40), "bath": [1, 2] * 20, "price": range(40)}), "h.csv")
    assert execute_tool("select_target", s, {"column": "price"})["success"]
    return s


@pytest.mark.parametrize("tool, message, allowed", [
    ("remove_outliers", "remove the outliers in price per sqft", True),
    ("remove_outliers", "what should I do next?", False),
    ("add_column", "create a bath per bhk ratio", True),
    ("add_column", "train a model", False),
    ("group_rare_categories", "too many locations, group the rare ones", True),
    ("drop_rows_where", "drop rows where bath > 10", True),
    ("drop_rows_where", "explain this chart", False),
    ("train_models", "what next?", True),  # ordinary steps are never filtered
])
def test_asked_for(tool, message, allowed):
    assert asked_for(tool, message) is allowed


def test_yes_to_a_proposal_counts_as_asking(session):
    session.memory += [{"role": "assistant", "content": "A few prices look like outliers. Remove them?"},
                       {"role": "user", "content": "yes do it"}]
    assert asked_for("remove_outliers", "yes do it", session)
    assert not asked_for("add_column", "yes do it", session)


def test_instructions_say_so():
    assert "ONLY WHEN THE USER ASKS" in STATIC_INSTRUCTIONS


def test_unrequested_judgement_calls_are_dropped_from_replies(session, monkeypatch):
    reply = {"response": "Next, a couple of ideas.", "suggested_actions": [
        {"tool": "remove_outliers", "arguments": {"value": "price"}, "explanation": ""},
        {"tool": "train_models", "arguments": {}, "explanation": ""},
    ]}

    async def ask(_session, _message):
        return {**reply, "suggested_actions": list(reply["suggested_actions"])}

    monkeypatch.setattr(copilot, "_providers", lambda: {"groq": ("Groq", ask, lambda: "m")})
    monkeypatch.setattr(copilot.settings, "AI_CHAIN", "groq")
    result = asyncio.run(copilot.get_copilot_response(session, "what should I do next?"))
    assert [a["tool"] for a in result["suggested_actions"]] == ["train_models"]
    result = asyncio.run(copilot.get_copilot_response(session, "remove outliers please"))
    assert [a["tool"] for a in result["suggested_actions"]] == ["remove_outliers", "train_models"]
