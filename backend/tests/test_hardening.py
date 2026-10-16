"""Fixes from the pre-deployment review: crashes, memory bombs, abuse limits, and wrong defaults."""

import io
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import app
from app.access import client_address
from app.config import settings
from app.session import DataSession, session_manager
from app.tools import execute_tool

H = {"X-Visitor-Id": "hardening-visitor-00001"}


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    with TestClient(app, headers=H) as c:
        yield c


def upload(client, df, name="d.csv"):
    r = client.post("/api/upload", files={"file": (name, df.to_csv(index=False).encode())})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def test_a_constant_column_does_not_crash_replies(client):
    """Its skewness is undefined (NaN), which JSON can't carry: replies used to crash."""
    df = pd.DataFrame({"same": [1.0] * 40, "x": range(40), "y": [0, 1] * 20})
    sid = upload(client, df)
    client.post("/api/action", json={"session_id": sid, "tool": "select_target", "arguments": {"column": "y"}})
    r = client.post("/api/action", json={"session_id": sid, "tool": "scale_all_features", "arguments": {}})
    step = r.json()["step"]["id"]
    r = client.put(f"/api/session/{sid}/recipe/{step}", json={"args": {}})
    assert r.status_code == 200
    same = next(c for c in r.json()["new_state_summary"]["columns"] if c["name"] == "same")
    assert same["skewness"] == 0.0


def test_onehot_refuses_id_like_columns():
    df = pd.DataFrame({"name": [f"person {i}" for i in range(100)], "city": ["a", "b"] * 50, "y": [0, 1] * 50})
    s = DataSession(df, "d.csv")
    run(s, "select_target", column="y")
    result = execute_tool("encode_all_categorical", s, {"method": "onehot"})
    assert not result["success"] and "'name'" in result["message"] and "label encoding" in result["message"]
    assert len(s.df.columns) == 3  # nothing changed
    run(s, "remove_columns", columns=["name"])
    run(s, "encode_all_categorical", method="onehot")  # fine once the ID-like column is gone


def test_rate_limits_use_the_address_our_proxy_saw():
    """Earlier X-Forwarded-For entries are written by the visitor and could be faked."""
    request = Request({"type": "http", "headers": [(b"x-forwarded-for", b"6.6.6.6, 1.2.3.4")], "client": ("10.0.0.1", 1)})
    assert client_address(request) == "1.2.3.4"


def test_oversized_requests_are_refused(client):
    r = client.put("/api/session/whatever/cells", content=b"x" * (3 * 1024 * 1024),
                   headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_chat_messages_have_a_length_limit(client):
    sid = upload(client, pd.DataFrame({"x": range(20), "y": [0, 1] * 10}))
    r = client.post("/api/chat", json={"session_id": sid, "message": "a" * 5000})
    assert r.status_code == 422


def test_models_download_instead_of_filling_the_server_disk(client, tmp_path):
    sid = upload(client, pd.DataFrame({"x": range(60), "y": [0, 1] * 30}))
    client.post("/api/action", json={"session_id": sid, "tool": "select_target", "arguments": {"column": "y"}})
    client.post("/api/action", json={"session_id": sid, "tool": "train_models", "arguments": {"algorithms": ["lr"]}})
    r = client.post("/api/action", json={"session_id": sid, "tool": "save_model", "arguments": {}})
    assert r.json()["success"] and "Export" in r.json()["message"]
    assert not os.path.exists(tmp_path / "models")

    r = client.get(f"/api/session/{sid}/export", params={"what": "model"})
    assert r.status_code == 200 and 'filename="d_lr_model.pkl"' in r.headers["content-disposition"]
    bundle = pickle.loads(r.content)
    assert bundle["target"] == "y" and bundle["model"].predict(pd.DataFrame({"x": [3]})).shape == (1,)


def test_only_recent_notebooks_stay_in_memory(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_SESSIONS_IN_MEMORY", 2)
    ids = [upload(client, pd.DataFrame({"x": range(10), "y": [0, 1] * 5})) for _ in range(3)]
    assert ids[0] not in session_manager._sessions and len(session_manager._sessions) == 2
    assert client.get(f"/api/session/{ids[0]}").status_code == 200  # reloads from storage


def _xy(seed=0, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": (x + rng.normal(scale=0.8, size=n) > 0).astype(int)})


def test_automl_winner_becomes_the_default():
    s = DataSession(_xy(), "d.csv")
    run(s, "select_target", column="y")
    result = run(s, "auto_ml")
    best = result["evidence"]["best_model"]
    assert "now the default" in result["message"]
    assert run(s, "predict", features={"x": 1.0, "noise": 0.0})["evidence"]["model_used"] == best


def test_automl_says_so_when_an_earlier_model_stays_the_default():
    from app.tools.model import best_model_name
    s = DataSession(_xy(), "d.csv")
    run(s, "select_target", column="y")
    run(s, "train_models", algorithms=["dt"])  # AutoML doesn't tune dt, so it keeps this result
    s.model_results["dt"]["cv_mean"] = 0.99  # pretend the earlier model did better than any tuned one
    result = run(s, "auto_ml")
    assert "'dt' still has the best practice score" in result["message"]
    assert best_model_name(s) == "dt"
    assert run(s, "predict", features={"x": 1.0, "noise": 0.0})["evidence"]["model_used"] == "dt"


def test_unknown_algorithms_are_named_and_long_names_work():
    s = DataSession(pd.DataFrame({"x": range(40), "y": [0, 1] * 20}), "d.csv")
    run(s, "select_target", column="y")
    assert "Available" in execute_tool("train_models", s, {"algorithms": ["tabnet"]})["message"]
    assert set(run(s, "train_models", algorithms=["Random Forest"])["evidence"]["results"][0]) >= {"model", "cv_mean"}


def test_a_blank_answer_does_not_turn_the_target_into_decimals():
    df = pd.DataFrame({"x": range(40), "y": [0.0, 1.0] * 20})
    df.loc[3, "y"] = np.nan
    s = DataSession(df, "d.csv")
    run(s, "select_target", column="y")
    run(s, "train_models", algorithms=["lr"])
    assert run(s, "predict", features={"x": 5})["evidence"]["prediction"] in ("0", "1")


def test_custom_views_can_say_df(monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", True)
    s = DataSession(pd.DataFrame({"x": range(20), "y": [0, 1] * 10}), "d.csv")
    run(s, "select_target", column="y")
    run(s, "custom_view", code="result = df.shape[0]")
