"""The notebook assistant: answers questions, or writes one code cell for the user to run."""

import asyncio

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app, assistant
from app.assistant import cell_kind, notebook_context, split_reply
from app.config import settings
from app.session import DataSession
from app.tools import execute_tool

H = {"X-Visitor-Id": "assistant-visitor-000001"}

CHANGE_CODE = '''import numpy as np
import pandas as pd

TRAIN_ONLY = True


def learn(train):
    return {}


def apply(df, learned):
    return df[df["sqft"] > 0]'''


@pytest.fixture
def session():
    s = DataSession(pd.DataFrame({"sqft": range(1, 41), "location": ["a", "b"] * 20, "price": range(40)}), "h.csv")
    assert execute_tool("select_target", s, {"column": "price"})["success"]
    return s


def fake_ai(monkeypatch, *answers):
    """The provider chain answers with these texts in turn; records what it was sent."""
    sent, queue = [], list(answers)

    async def complete(system, messages):
        sent.append(messages)
        return {"text": queue.pop(0), "provider": "Groq", "model": "m"}

    monkeypatch.setattr(assistant, "complete", complete)
    return sent


def test_split_reply_and_kinds():
    words, code = split_reply("Removes bad rows.\n```python\n" + CHANGE_CODE + "\n```\nThen train.")
    assert words == "Removes bad rows.\n\nThen train." and code.startswith("import numpy")
    assert cell_kind(CHANGE_CODE) == "change"
    assert cell_kind("from sklearn.linear_model import Ridge\nmodels = {'ridge': Ridge()}") == "models"
    assert cell_kind("result = train.describe()") == "look"
    assert split_reply("Just an answer.") == ("Just an answer.", None)


def test_context_has_the_recipe_mode_and_latest_cells(session):
    execute_tool("remove_columns", session, {"columns": ["location"]})
    session.cells = [{"type": "run", "tool": "custom_view", "message": "ok", "args": {"code": "result = train.shape"}}]
    context = notebook_context(session)
    assert "predicting 'price'" in context and "remove_columns(columns=location)" in context
    assert "result = train.shape" in context and "sqft" in context


def test_it_remembers_the_notebooks_code_names_and_requests(session):
    """Consistency: every earlier cell (with the user's edits), the names it uses, what was asked."""
    session.cells = [
        {"type": "ask", "prompt": "price per sqft by location"},
        {"type": "run", "tool": "custom_view", "message": "ok",
         "args": {"code": "pps = train['price'] / train['sqft']\ntop = pps.groupby(train['location']).mean()\nresult = top"}},
        {"type": "run", "tool": "impute_missing", "args": {"column": "sqft", "strategy": "median"}, "message": "Filled 0"},
        {"type": "code", "kind": "change", "source": CHANGE_CODE.replace("return {}", "return {'min_sqft': 1}")},
    ]
    context = notebook_context(session)
    assert "- price per sqft by location" in context  # what they asked for
    assert "pps, top" in context and '"min_sqft"' in context  # names to reuse
    assert "[2] look cell (ran)" in context and "pps = train['price']" in context  # the code itself
    assert "[3] built-in impute_missing(column=sqft, strategy=median)" in context
    assert "[4] change cell (not run yet)" in context


def test_old_code_past_the_budget_is_summarised(session, monkeypatch):
    monkeypatch.setattr(assistant, "CODE_BUDGET", 60)
    session.cells = [{"type": "run", "tool": "custom_view", "args": {"code": "old_name = 1\nresult = old_name"}},
                     {"type": "run", "tool": "custom_view", "args": {"code": "x = train.shape\nresult = x" + " " * 30}}]
    context = notebook_context(session)
    assert "[1] look cell (ran), 2 lines; defines: old_name" in context and "x = train.shape" in context


def test_the_conversation_keeps_eight_exchanges(session, monkeypatch):
    session.memory = [{"role": r, "content": f"{r} {i}", "kind": "assist"} for i in range(12) for r in ("user", "assistant")]
    sent = fake_ai(monkeypatch, "Sure.")
    asyncio.run(assistant.assist(session, "and now?"))
    history = sent[0][:-1]
    assert len(history) == 16 and history[0]["content"] == "user 4"


def test_writes_code_for_a_request(session, monkeypatch):
    fake_ai(monkeypatch, "Drops rows with no area.\n```python\n" + CHANGE_CODE + "\n```")
    result = asyncio.run(assistant.assist(session, "drop rows with zero sqft"))
    assert result["kind"] == "change" and result["code"].startswith("import numpy") and not result["problem"]
    assert result["reply"] == "Drops rows with no area."


def test_answers_questions_without_code(session, monkeypatch):
    fake_ai(monkeypatch, "Your recipe has one step so far.")
    result = asyncio.run(assistant.assist(session, "what have I done?"))
    assert "code" not in result and result["reply"] == "Your recipe has one step so far."


def test_broken_code_is_retried_once(session, monkeypatch):
    sent = fake_ai(monkeypatch, "```python\nresult = train.head(\n```", "Fixed.\n```python\nresult = train.head()\n```")
    result = asyncio.run(assistant.assist(session, "show the first rows"))
    assert result["code"] == "result = train.head()" and len(sent) == 2
    assert "syntax error" in sent[1][-1]["content"]


def test_the_code_it_writes_runs_as_a_cell(session, monkeypatch):
    """What the assistant writes goes through the same cells as hand-written code."""
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", True)
    fake_ai(monkeypatch, "```python\n" + CHANGE_CODE + "\n```")
    result = asyncio.run(assistant.assist(session, "drop rows with zero sqft"))
    run = execute_tool("custom_code", session, {"code": result["code"]})
    assert run["success"], run["message"]


def test_endpoint_remembers_the_conversation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    fake_ai(monkeypatch, "Here.\n```python\nresult = train.describe()\n```", "It summarises every column.")
    with TestClient(app, headers=H) as client:
        csv = "x,y\n" + "\n".join(f"{i},{i % 2}" for i in range(20))
        sid = client.post("/api/upload", files={"file": ("d.csv", csv.encode())}).json()["session_id"]
        client.post("/api/action", json={"session_id": sid, "tool": "clean_only", "arguments": {}})
        r = client.post("/api/assist", json={"session_id": sid, "message": "describe the data"}).json()
        assert r["kind"] == "look" and r["code"] == "result = train.describe()"
        r = client.post("/api/assist", json={"session_id": sid, "message": "what does that do?"}).json()
        assert r["reply"] == "It summarises every column."


def test_when_no_provider_answers(session, monkeypatch):
    async def complete(system, messages):
        return {"failed": True, "text": "The AI couldn't answer right now: Groq was unavailable (quota)."}
    monkeypatch.setattr(assistant, "complete", complete)
    result = asyncio.run(assistant.assist(session, "anything"))
    assert result["failed"] and "unavailable" in result["reply"]
