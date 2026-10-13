"""The code sandbox: user code never runs in the app, can't run forever, can't see the app's secrets,
and nothing it sends back can run code in the app. The Linux protections (no network, no starting
programs, memory limits, another user) are tested on Linux (CI); the rest everywhere."""

import base64
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app import sandbox
from app import sandbox_runner as runner
from app.config import settings
from app.session import DataSession
from app.tools import execute_tool

LINUX = sys.platform.startswith("linux")


@pytest.fixture(autouse=True)
def fresh_sandbox(monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", True)
    monkeypatch.setattr(settings, "CODE_SANDBOX", "local")
    sandbox.reset()
    yield
    sandbox.reset()


@pytest.fixture
def session():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"sqft": rng.uniform(500, 3000, 120), "bath": rng.integers(1, 4, 120).astype(float)})
    df["price"] = df["sqft"] * 0.05 + df["bath"] * 3 + rng.normal(0, 2, 120)
    s = DataSession(df, "houses.csv")
    assert execute_tool("select_target", s, {"column": "price"})["success"]
    return s


def view(session, code):
    return execute_tool("custom_view", session, {"code": code})


# ── Never in the app ─────────────────────────────────────────────────────────

def test_code_runs_in_another_process(session):
    result = view(session, "import os\nresult = os.getpid()")
    assert result["success"] and result["evidence"]["output"] != str(os.getpid())


def test_the_apps_secrets_are_not_there(session, monkeypatch):
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sk-very-secret")
    sandbox.reset()  # a new worker, started while the app has the secret
    result = view(session, "import os\nresult = repr(os.environ.get('SUPABASE_SECRET_KEY'))")
    assert result["evidence"]["output"] == "None"


def test_an_endless_loop_is_stopped_and_the_next_cell_works(session, monkeypatch):
    monkeypatch.setattr(settings, "CODE_TIME_LIMIT_SECONDS", 2)
    stuck = view(session, "while True:\n    pass")
    assert not stuck["success"] and "ran longer than 2 seconds" in stuck["message"]
    assert view(session, "result = len(train)")["evidence"]["output"] == "96"


def test_a_crash_in_the_worker_is_survived(session):
    crashed = view(session, "import os\nos._exit(3)")
    assert not crashed["success"]
    assert view(session, "result = 1 + 1")["evidence"]["output"] == "2"


# ── Nothing that comes back can run code ─────────────────────────────────────

def test_an_answer_with_pickled_data_is_refused():
    import pickle
    hostile = json.dumps({"ok": True, "frame": {"__pickle__": base64.b64encode(pickle.dumps("x")).decode()}}).encode()
    assert sandbox._frames(hostile) == {"ok": False, "error": "The sandbox sent back something unreadable."}


def _skops(model) -> str:
    from skops.io import dumps
    return base64.b64encode(dumps(model)).decode()


def test_scikit_learn_models_load():
    X, y = np.random.rand(30, 2), np.random.rand(30)
    for model in (RandomForestRegressor(n_estimators=3, random_state=0),
                  make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), RandomForestRegressor(n_estimators=2))):
        model.fit(X, y)
        assert np.allclose(sandbox.load_model(_skops(model)).predict(X), model.predict(X))


def test_a_model_carrying_a_dangerous_function_is_refused():
    model = FunctionTransformer(func=os.system)  # would run a shell command on transform
    with pytest.raises(ValueError, match="scikit-learn parts"):
        sandbox.load_model(_skops(model))


# ── Every kind of cell through the sandbox ───────────────────────────────────

CHANGE = """
TRAIN_ONLY = False

def learn(train):
    return {"median": train["sqft"].median()}

def apply(df, learned):
    df["big"] = (df["sqft"] > learned["median"]).astype(int)
    return df
"""


def test_a_change_cell_learns_applies_and_replays_on_test_rows(session):
    result = execute_tool("custom_code", session, {"code": CHANGE})
    assert result["success"], result["message"]
    assert "big" in session.df and "big" in session.prepare(session.test_df)
    assert session.recipe[-1].learned["median"] == pytest.approx(session.raw_train["sqft"].median())


def test_train_only_is_read_without_running_the_code(session):
    code = CHANGE.replace("TRAIN_ONLY = False", "TRAIN_ONLY = 1 + 1")
    result = execute_tool("custom_code", session, {"code": code})
    assert not result["success"] and "TRAIN_ONLY must be set to True or False" in result["message"]


def test_custom_models_train_in_the_sandbox_and_take_the_exam(session):
    code = "from sklearn.linear_model import Ridge\nmodels = {'ridge': Ridge(alpha=3)}"
    trained = execute_tool("custom_models", session, {"code": code})
    assert trained["success"], trained["message"]
    assert "ridge" in session.trained_models and trained["evidence"]["results"][0]["cv_mean"] > 0.5
    assert execute_tool("final_exam", session, {"algorithm": "ridge"})["success"]
    assert execute_tool("predict", session, {"features": {"sqft": 1500, "bath": 2}})["success"]


# ── Who may run code ──────────────────────────────────────────────────────────

def test_accounts_only(session, monkeypatch):
    monkeypatch.setattr(settings, "CODE_FOR", "accounts")
    refused = view(session, "result = 1")
    assert not refused["success"] and "needs an account" in refused["message"]
    session.owner = "user:abc"
    assert view(session, "result = 1")["success"]


def test_hardened_without_its_server_falls_back_or_stays_off(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CODE_SANDBOX", "hardened")
    monkeypatch.setattr(settings, "SANDBOX_SOCKET", str(tmp_path / "missing.sock"))
    monkeypatch.setattr(settings, "MODAL_TOKEN_ID", "")
    assert sandbox.status() == {"enabled": False, "reason": f"the sandbox server isn't running (no socket at {tmp_path / 'missing.sock'})"}
    sandbox.reset()
    monkeypatch.setattr(settings, "MODAL_TOKEN_ID", "id")
    monkeypatch.setattr(settings, "MODAL_TOKEN_SECRET", "secret")
    assert sandbox.status()["sandbox"] == "modal"


# ── Linux: the hardening itself ───────────────────────────────────────────────

def _probe(memory_mb=256):
    job = runner.encode({"kind": "probe", "app_pid": os.getpid(), "app_uid": os.getuid(), "data_dir": "/nonexistent",
                         "memory_mb": memory_mb, "limits": {"seconds": 30, "memory_mb": memory_mb}}, exact_frames=True)
    return json.loads(runner.run_forked(job, harden_it=True))


@pytest.mark.skipif(not LINUX, reason="seccomp and resource limits are Linux features")
def test_hardened_child_has_no_network_no_programs_and_a_memory_cap():
    result = _probe()
    assert result["ok"], result
    stopped = result["stopped"]
    assert stopped["network"] and stopped["programs"] and stopped["memory"]


@pytest.mark.skipif(not LINUX, reason="Linux only")
def test_hardened_child_cannot_open_a_socket_from_user_code():
    job = runner.encode({"kind": "view", "limits": {"seconds": 30, "memory_mb": 1024},
                         "code": "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)\nresult = 'connected'",
                         "frames": {"train": pd.DataFrame({"a": [1]})}}, exact_frames=True)
    result = json.loads(runner.run_forked(job, harden_it=True))
    assert not result["ok"] and ("PermissionError" in result["error"] or "Errno 13" in result["error"])


@pytest.mark.skipif(not LINUX or os.geteuid() != 0, reason="needs root: dropping to another user")
def test_as_root_the_child_becomes_another_user():
    result = _probe()
    assert result["stopped"]["other_user"] and result["stopped"]["app_secrets"]
