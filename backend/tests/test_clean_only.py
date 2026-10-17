"""Clean-only mode: cleaning data without a model (e.g. a customer call list), using the steps of a
real notebook: duplicates, a useless column, opted-out customers, missing phone numbers."""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from app.config import settings
from app.copilot_prompt import state_snapshot, tool_specs_for
from app.export import pipeline_script
from app.session import DataSession, session_manager
from app.tools import execute_tool

H = {"X-Visitor-Id": "clean-only-visitor-00001"}


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


def call_list():
    df = pd.DataFrame({
        "CustomerID": range(1, 21),
        "Last_Name": ["Smith", "Doe_", "/Lee", "Ray..", "Kim"] * 4,
        "Phone_Number": ["123-545-5421", "(876) 678 3469", None, "N/a", "7066950392"] * 4,
        "Do_Not_Contact": ["No", "Yes", "No", None, "No"] * 4,
        "Not_Useful_Column": [True] * 20,
    })
    return pd.concat([df, df.iloc[:3]], ignore_index=True)  # 3 duplicate rows


@pytest.fixture
def session():
    s = DataSession(call_list(), "Customer Call List.xlsx")
    run(s, "clean_only")
    return s


def test_works_on_all_rows_with_statistics_visible(session):
    assert len(session.df) == 23 and not session.is_split and session.test_df is None
    summary = session.get_summary()
    assert summary["clean_only"] and "missing_ratio" in summary["columns"][2]


def test_the_notebooks_cleaning(session):
    run(session, "remove_duplicates")                                      # all rows, not "training rows"
    run(session, "remove_columns", columns=["Not_Useful_Column"])
    run(session, "drop_rows_where", conditions='Do_Not_Contact == "Yes"')
    run(session, "drop_missing_rows", columns=["Phone_Number"])
    # 23 - 3 duplicates - 4 opted out - 4 blank phones. The 4 "N/a" phones are text, not blanks, so
    # they stay (turning "N/a" into a blank is the "replace values" text step, not built yet)
    assert len(session.df) == 12
    assert "Yes" not in set(session.df["Do_Not_Contact"]) and session.df["Phone_Number"].notna().all()


def test_messages_speak_of_rows_not_training_rows(session):
    """There are no training or test rows here, so the steps don't talk about them."""
    message = run(session, "drop_rows_where", conditions='Do_Not_Contact == "Yes"')["message"]
    assert message.startswith("Dropped 5 rows where") and "training" not in message and "Test rows" not in message
    # With a target, the full explanation is still there
    s = DataSession(call_list(), "c.csv")
    run(s, "select_target", column="Do_Not_Contact")
    assert "training rows" in run(s, "drop_rows_where", conditions="CustomerID > 18")["message"]


def test_no_model_tools(session):
    for tool in ("train_models", "final_exam", "predict"):
        result = execute_tool(tool, session, {})
        assert not result["success"] and "clean-only mode" in result["message"]
    names = {t["name"] for t in tool_specs_for(session)}
    assert "remove_duplicates" in names and "train_models" not in names and "select_target" in names
    assert "CLEAN-ONLY MODE" in state_snapshot(session)


def test_choosing_a_target_later_stays_leak_free():
    df = pd.DataFrame({"x": np.arange(100.0), "y": [0, 1] * 50})
    df.loc[[3, 7, 50], "x"] = np.nan
    s = DataSession(df, "d.csv")
    run(s, "clean_only")
    run(s, "impute_all_missing")            # learned from ALL rows while cleaning
    run(s, "select_target", column="y")     # now: lock test rows from the original data
    assert s.is_split and not s.clean_only
    # The fill value is re-learned from the training rows only
    assert s.recipe[0].learned["fill"]["x"] == pytest.approx(s.raw_train["x"].mean(), abs=1e-6)
    assert len(s.df) + len(s.test_df) == 100


def test_exports_the_whole_table_and_a_cleaning_script(session, tmp_path, monkeypatch):
    run(session, "remove_duplicates")
    run(session, "drop_rows_where", conditions='Do_Not_Contact == "Yes"')
    monkeypatch.chdir(tmp_path)
    session.original_df.to_csv("calls.csv", index=False)
    scope = {"__name__": "pipeline"}
    exec(compile(pipeline_script(session), "pipeline.py", "exec"), scope)
    cleaned = scope["main"]("calls.csv")
    pd.testing.assert_frame_equal(cleaned, session.df, check_dtype=False)


def test_api_round_trip_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    with TestClient(app, headers=H) as client:
        buf = io.BytesIO()
        call_list().to_excel(buf, index=False)
        sid = client.post("/api/upload", files={"file": ("calls.xlsx", buf.getvalue())}).json()["session_id"]
        act = lambda tool, **a: client.post("/api/action", json={"session_id": sid, "tool": tool, "arguments": a}).json()
        assert act("clean_only")["new_state_summary"]["clean_only"]
        act("remove_duplicates")
        r = client.get(f"/api/session/{sid}/export", params={"what": "train", "format": "csv"})
        assert len(pd.read_csv(io.BytesIO(r.content))) == 20  # every row, cleaned
        assert client.get(f"/api/session/{sid}/export", params={"what": "script"}).status_code == 200

        session_manager._sessions.clear()  # a fresh server rebuilds it from storage
        summary = client.get(f"/api/session/{sid}").json()["summary"]
        assert summary["clean_only"] and summary["shape"][0] == 20


def test_not_after_a_target_is_chosen():
    s = DataSession(pd.DataFrame({"x": range(20), "y": [0, 1] * 10}), "d.csv")
    run(s, "select_target", column="y")
    assert "already predicts" in execute_tool("clean_only", s, {})["message"]


# ── No forced choice (the sealed envelope is tested in test_sealed_envelope.py) ───────

def test_a_small_file_explores_all_rows_and_says_why():
    s = DataSession(call_list(), "Customer Call List.xlsx")  # 23 rows: too few to seal
    result = execute_tool("remove_duplicates", s, {})
    assert result["success"] and result["started_exploring"] and result["message"].startswith("(Exploring all 23 rows")
    assert s.clean_only and s.explored_first and s.sealed is None and len(s.df) == 20
    assert "Models need a column to predict" in execute_tool("train_models", DataSession(call_list(), "c.csv"), {})["message"]


def test_explore_and_clean_then_predicting_carries_the_note():
    """Explore & clean shows every row (no envelope): a later final exam says so. Choosing the
    target first has no such note."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=120), "z": rng.normal(size=120)})
    df["y"] = df.x * 2 + rng.normal(size=120)
    s = DataSession(df, "e.csv")
    run(s, "clean_only")
    run(s, "select_target", column="y")
    run(s, "train_models", algorithms=["linear"])
    exam = run(s, "final_exam")
    assert "explored before they were locked away" in exam["message"] and exam["evidence"]["exam"]["explored_first"]
    fresh = DataSession(df, "e.csv")
    run(fresh, "select_target", column="y")
    run(fresh, "train_models", algorithms=["linear"])
    assert "explored before" not in run(fresh, "final_exam")["message"]
