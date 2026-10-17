"""The sealed envelope: explore first, predict later, and the score is still honest.

When someone starts exploring without choosing a column to predict, a share of the rows is sealed:
never shown (charts, statistics, code cells, the assistant, exports of the visible rows). If they later
predict a column, the sealed rows are its test set, on every re-lock too. Exporting all rows opens the
envelope, and that's remembered."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app import app, sandbox
from app.assistant import notebook_context
from app.config import settings
from app.session import DataSession, session_manager
from app.tools import execute_tool

H = {"X-Visitor-Id": "sealed-envelope-visitor-01"}


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


def table(n=400):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"row_id": range(n), "x": rng.normal(size=n), "z": rng.normal(size=n),
                       "day": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d")})
    df["y"] = df.x * 3 + rng.normal(size=n)
    return df


def sealed_ids(s):
    return set(s.original_df.loc[s.sealed["rows"], "row_id"])


def test_exploring_never_shows_the_sealed_rows(monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", True)
    monkeypatch.setattr(settings, "CODE_SANDBOX", "local")
    sandbox.reset()
    s = DataSession(table(), "t.csv")
    first = run(s, "plot_histogram", column="x")
    assert "Exploring 320 of 400 rows. 80 rows (20%, chosen at random) are sealed" in first["message"]
    hidden = sealed_ids(s)
    assert len(hidden) == 80 and not hidden & set(s.df["row_id"])
    assert s.get_summary()["sealed_rows"] == 80 and s.get_summary()["shape"][0] == 320
    shown = run(s, "custom_view", code="result = len(set(train['row_id']) & set(SEALED))".replace(
        "SEALED", repr(sorted(hidden))))["evidence"]["output"]
    assert shown == "0"  # a code cell sees none of the sealed rows
    assert "80 sealed" in notebook_context(s)
    sandbox.reset()


def test_predicting_later_tests_on_exactly_the_sealed_rows_with_no_note():
    s = DataSession(table(), "t.csv")
    run(s, "start_exploring", share=25)
    hidden = sealed_ids(s)
    run(s, "remove_columns", columns=["day"])
    locked = run(s, "select_target", column="y", test_size=0.4)["message"]
    assert "rows sealed when you started exploring" in locked and "You asked for 40%" in locked
    assert set(s.test_df["row_id"]) == hidden and not hidden & set(s.df["row_id"])
    run(s, "train_models", algorithms=["linear"])
    assert "explored before" not in run(s, "final_exam")["message"]
    # Changing the target re-locks: the same sealed rows, never a reshuffle that would expose them
    run(s, "select_target", column="z")
    assert set(s.test_df["row_id"]) == hidden


def test_sealing_the_most_recent_rows_keeps_a_time_split_honest():
    s = DataSession(table(), "t.csv")
    assert s.get_summary()["date_columns"] == ["day"]
    run(s, "start_exploring", share=0.2, seal_by="day")
    assert sealed_ids(s) == set(range(320, 400))  # the latest 80 days
    run(s, "select_target", column="y", split="time", time_column="day")
    assert set(s.test_df["row_id"]) == set(range(320, 400))


def test_a_time_split_on_a_random_seal_is_refused():
    s = DataSession(table(), "t.csv")
    run(s, "start_exploring")
    refused = execute_tool("select_target", s, {"column": "y", "split": "time", "time_column": "day"})
    assert not refused["success"] and "chosen at random" in refused["message"]


def test_exporting_all_rows_opens_the_envelope_and_it_stays_open(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    with TestClient(app, headers=H) as client:
        sid = client.post("/api/upload", files={"file": ("t.csv", table().to_csv(index=False).encode())}).json()["session_id"]
        client.post("/api/action", json={"session_id": sid, "tool": "start_exploring", "arguments": {"share": 0.2}})
        visible = client.get(f"/api/session/{sid}/export", params={"what": "train"}).text
        assert len(visible.strip().splitlines()) == 321  # header + the 320 visible rows
        everything = client.get(f"/api/session/{sid}/export", params={"what": "all"}).text
        assert len(everything.strip().splitlines()) == 401
        session_manager._sessions.pop(sid, None)  # reload from storage
        summary = client.get(f"/api/session/{sid}").json()["summary"]
        assert summary["sealed_rows"] == 0 and summary["explored_first"]


def test_the_seal_survives_a_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    with TestClient(app, headers=H) as client:
        sid = client.post("/api/upload", files={"file": ("t.csv", table().to_csv(index=False).encode())}).json()["session_id"]
        client.post("/api/action", json={"session_id": sid, "tool": "start_exploring", "arguments": {"share": 0.3}})
        before = session_manager.get_session(sid).sealed["rows"]
        session_manager._sessions.pop(sid, None)
        again = session_manager.get_session(sid)
        assert again.sealed["rows"] == before and len(again.df) == 280
