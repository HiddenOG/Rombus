"""Custom code steps: they run, replay, report errors clearly, and get checked for peeking."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from app.code_checks import peeking_warnings
from app.config import settings
from app.recipe import CUSTOM_TEMPLATE, Context, Line, apply_to, replay
from app.session import DataSession
from app.tools import execute_tool

CTX = Context(target="label")
TRAIN = pd.DataFrame({"age": [20.0, None, 40.0], "label": [0, 1, 0]})
TEST = pd.DataFrame({"age": [None, 99.0], "label": [1, 1]})

HONEST = '''
def learn(train):
    return {"middle": train["age"].median()}

def apply(df, learned):
    df["age"] = df["age"].fillna(learned["middle"])
    return df
'''

PEEKING = '''
def learn(train):
    return {}

def apply(df, learned):
    df["age"] = df["age"].fillna(df["age"].median())  # computed from whatever rows it's given!
    return df
'''


def custom(code):
    return Line("custom_code", {"code": code})


class TestRunning:
    def test_learns_from_training_and_reuses_on_test(self):
        line = custom(HONEST)
        cleaned = replay(TRAIN, [line], CTX)
        assert line.learned == {"middle": 30.0}
        assert cleaned["age"].tolist() == [20.0, 30.0, 40.0]
        assert apply_to(TEST, [line], CTX)["age"].tolist() == [30.0, 99.0]

    def test_template_runs_unchanged(self):
        pd.testing.assert_frame_equal(replay(TRAIN, [custom(CUSTOM_TEMPLATE)], CTX), TRAIN)

    def test_train_only_flag(self):
        code = "TRAIN_ONLY = True\ndef learn(train): return {}\ndef apply(df, learned): return df.head(1)"
        line = custom(code)
        assert len(replay(TRAIN, [line], CTX)) == 1
        assert len(apply_to(TEST, [line], CTX)) == 2  # skipped for test rows

    def test_learned_values_made_plain(self):
        code = "def learn(train): return {'means': train.mean(numeric_only=True), 'n': np.int64(3)}\n" \
               "def apply(df, learned): return df"
        line = custom(code)
        replay(TRAIN, [line], CTX)
        assert line.learned == {"means": {"age": 30.0, "label": pytest.approx(1 / 3)}, "n": 3}


class TestErrors:
    @pytest.mark.parametrize("code,message", [
        ("def learn(train) return {}", "syntax error on line 1"),
        ("def learn(train): return {}", "must define a function called apply()"),
        ("def learn(train): return 5\ndef apply(df, l): return df", "learn() must return a dict"),
        ("def learn(train): return {}\ndef apply(df, l): return 1", "apply() must return the rows as a DataFrame"),
        ("def learn(train):\n    return {}\ndef apply(df, l):\n    return df[missing]", "NameError: name 'missing' is not defined (line 4)"),
    ])
    def test_clear_messages(self, code, message):
        with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
            replay(TRAIN, [custom(code)], CTX)

    def test_switched_off(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", False)
        with pytest.raises(ValueError, match="switched off"):
            replay(TRAIN, [custom(HONEST)], CTX)


class TestPeekingCheck:
    def test_flags_statistics_in_apply(self):
        warnings = peeking_warnings(PEEKING)
        assert len(warnings) == 1
        assert "median()" in warnings[0] and "Line 6" in warnings[0]

    def test_statistics_in_learn_are_fine(self):
        assert peeking_warnings(HONEST) == []

    def test_other_variables_not_flagged(self):
        code = "def apply(df, learned):\n    ref = learned['table']\n    df['x'] = ref.max()\n    return df"
        assert peeking_warnings(code) == []


class TestConvertAndRevert:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
        with TestClient(app) as c:
            yield c

    def test_edit_builtin_code_then_revert(self, client):
        csv = b"age,label\n" + b"\n".join(f"{'' if i % 4 == 0 else 20 + i},{i % 2}".encode() for i in range(20))
        sid = client.post("/api/upload", files={"file": ("d.csv", csv, "text/csv")}).json()["session_id"]

        def act(tool, **args):
            return client.post("/api/action", json={"session_id": sid, "tool": tool, "arguments": args}).json()

        act("select_target", column="label")
        step = act("impute_missing", column="age", strategy="median")["step"]
        assert "train['age'].median()" in step["editable_code"]
        assert "What learn() returned" not in step["editable_code"]

        edited = client.put(f"/api/session/{sid}/recipe/{step['id']}/code", json={"code": PEEKING})
        assert edited.status_code == 200
        view = edited.json()["new_state_summary"]["recipe"][0]
        assert view["tool"] == "custom_code" and view["id"] == step["id"]
        assert view["based_on"]["tool"] == "impute_missing"
        assert view["warnings"]  # peeking flagged

        broken = client.put(f"/api/session/{sid}/recipe/{step['id']}/code", json={"code": "oops("})
        assert broken.status_code == 400 and "syntax error" in broken.json()["detail"]

        reverted = client.post(f"/api/session/{sid}/recipe/{step['id']}/revert")
        assert reverted.json()["new_state_summary"]["recipe"][0]["tool"] == "impute_missing"

    def test_custom_step_via_action(self, client):
        csv = b"age,label\n" + b"\n".join(f"{20 + i},{i % 2}".encode() for i in range(20))
        sid = client.post("/api/upload", files={"file": ("d.csv", csv, "text/csv")}).json()["session_id"]
        client.post("/api/action", json={"session_id": sid, "tool": "select_target", "arguments": {"column": "label"}})
        result = client.post("/api/action", json={"session_id": sid, "tool": "custom_code", "arguments": {"code": HONEST}}).json()
        assert result["success"], result["message"]
        assert result["step"]["kind"] == "custom"
