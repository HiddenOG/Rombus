"""Fixes from a real session: a misleading target error, an AI going in circles, a stage tracker
that didn't follow the work, and undoing several cells at once."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from app.config import settings
from app.copilot_prompt import state_snapshot
from app.session import DataSession
from app.tools import execute_tool

H = {"X-Visitor-Id": "real-use-visitor-000001"}


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


@pytest.fixture
def session():
    rng = np.random.default_rng(4)
    n = 120
    df = pd.DataFrame({"total_sqft": [f"{v}" for v in rng.integers(500, 3000, n)],
                       "bath": rng.integers(1, 4, n), "location": rng.choice(["a", "b"], n),
                       "price": rng.integers(20, 300, n)})
    s = DataSession(df, "houses.csv")
    run(s, "select_target", column="price")
    return s


def test_formula_with_the_target_says_why(session):
    """Seen live: 'price / total_sqft' said "Column 'price' not found" while price was right there."""
    run(session, "clean_numeric_string", column="total_sqft")
    message = execute_tool("add_column", session, {"name": "price_per_sqft", "formula": "price / total_sqft"})["message"]
    assert "column you're predicting" in message and "not found" not in message


def test_ai_sees_each_step_with_its_settings(session):
    """Seen live: the AI kept offering to clean total_sqft after it had been cleaned."""
    run(session, "clean_numeric_string", column="total_sqft")
    run(session, "remove_columns", columns=["location"])
    snapshot = state_snapshot(session)
    assert "clean_numeric_string(column=total_sqft) → remove_columns(columns=location)" in snapshot


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    with TestClient(app, headers=H) as c:
        yield c


def notebook(client):
    csv = "total_sqft,bath,location,price\n" + "\n".join(f"{600 + i}sqft,{1 + i % 3},{'ab'[i % 2]},{50 + i}" for i in range(60))
    sid = client.post("/api/upload", files={"file": ("h.csv", csv.encode())}).json()["session_id"]
    act = lambda tool, **a: client.post("/api/action", json={"session_id": sid, "tool": tool, "arguments": a}).json()
    act("select_target", column="price")
    return sid, act


def test_stage_follows_the_work_and_never_goes_back(client):
    sid, act = notebook(client)
    assert act("remove_duplicates")["new_state_summary"]["stage"] == "SCRUB"
    assert act("plot_histogram", column="bath")["new_state_summary"]["stage"] == "EXPLORE"
    assert act("encode_all_categorical", method="label")["new_state_summary"]["stage"] == "EXPLORE"  # not back
    assert act("train_models", algorithms=["rf"])["new_state_summary"]["stage"] == "MODEL"
    assert act("explain_model")["new_state_summary"]["stage"] == "INTERPRET"


def test_undo_several_steps_at_once(client):
    sid, act = notebook(client)
    ids = [act(tool, **args)["step"]["id"] for tool, args in
           [("remove_duplicates", {}), ("encode_all_categorical", {"method": "label"}), ("scale_all_features", {})]]
    r = client.post(f"/api/session/{sid}/recipe/delete", json={"step_ids": ids[:2]})
    assert r.status_code == 200 and "Removed 2 steps" in r.json()["message"]
    assert [s["tool"] for s in r.json()["new_state_summary"]["recipe"]] == ["scale_all_features"]


def test_undo_several_is_all_or_nothing(client):
    sid, act = notebook(client)
    clean = act("clean_numeric_string", column="total_sqft")["step"]["id"]  # "612sqft" -> 612
    act("add_column", name="per_bath", formula="total_sqft / bath")
    dup = act("remove_duplicates")["step"]["id"]
    # Undoing the cleaning would break the formula after it: nothing changes, and it says why
    r = client.post(f"/api/session/{sid}/recipe/delete", json={"step_ids": [dup, clean]})
    assert r.status_code == 400 and "break a later step" in r.json()["detail"]
    recipe = client.get(f"/api/session/{sid}/recipe").json()["recipe"]
    assert [s["tool"] for s in recipe] == ["clean_numeric_string", "add_column", "remove_duplicates"]
