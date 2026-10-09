"""Recipe inside a session: edit/delete any step, models cleared on change, saved across restarts."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from app.config import settings
from app.session import DataSession, SessionManager
from app.tools import execute_tool


@pytest.fixture
def session():
    df = pd.DataFrame({
        "age": [20.0, None, 40.0, 50.0, None, 60.0, 30.0, 45.0, 35.0, 25.0],
        "city": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "income": ["$1,000", "$2,000", "$3,000", "$4,000", "$5,000", "$6,000", "$7,000", "$8,000", "$9,000", "$1,500"],
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    s = DataSession(df, "people.csv")
    assert execute_tool("select_target", s, {"column": "label"})["success"]
    return s


class TestEditAnyStep:
    def test_changing_an_early_step_replays_later_ones(self, session):
        session.add_step("impute_missing", {"column": "age", "strategy": "mean"})
        session.add_step("scale_all_features", {})
        mean_note = session.recipe[1].learned["scale"]["age"]["mean"]

        session.update_step(0, {"column": "age", "strategy": "median"})

        assert session.recipe[0].args["strategy"] == "median"
        # Step 2 re-learned its sticky note from the new data
        assert session.recipe[1].learned["scale"]["age"]["mean"] != mean_note

    def test_delete_a_middle_step(self, session):
        session.add_step("remove_columns", {"columns": ["city"]})
        session.add_step("impute_missing", {"column": "age"})
        session.add_step("clean_numeric_string", {"column": "income"})
        session.delete_step(0)
        assert [l.tool for l in session.recipe] == ["impute_missing", "clean_numeric_string"]
        assert "city" in session.df.columns

    def test_delete_that_breaks_a_later_step_is_refused(self, session):
        session.add_step("clean_numeric_string", {"column": "income"})
        session.add_step("log_transform", {"column": "income"})  # needs income to be numeric
        before = session.df.copy()
        with pytest.raises(ValueError, match=r"break a later step: 'log transform'"):
            session.delete_step(0)
        assert len(session.recipe) == 2
        pd.testing.assert_frame_equal(session.df, before)

    def test_bad_edit_is_refused(self, session):
        session.add_step("impute_missing", {"column": "age"})
        with pytest.raises(ValueError, match="Unknown strategy"):
            session.update_step(0, {"column": "age", "strategy": "magic"})
        assert session.recipe[0].args == {"column": "age"}

    def test_no_such_step(self, session):
        with pytest.raises(ValueError, match="no step 3"):
            session.delete_step(2)


class TestModelsClearedOnChange:
    def test_cleaning_after_training_clears_models(self, session):
        session.add_step("impute_missing", {"column": "age"})
        session.add_step("remove_columns", {"columns": ["income"]})
        execute_tool("train_test_split", session, {"test_size": 0.3})
        execute_tool("train_models", session, {"algorithms": ["dt"], "cv_folds": 2})
        assert session.trained_models

        session.add_step("scale_all_features", {})
        assert session.trained_models == {}
        assert session.X_train is None
        assert session.models_cleared


class TestSavedAcrossRestarts:
    def test_recipe_rebuilt_on_load(self, session, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
        session.add_step("impute_missing", {"column": "age", "strategy": "median"})
        session.add_step("encode_categorical", {"column": "city", "method": "onehot"})
        manager = SessionManager()
        manager.initialize()
        manager.save(session)

        restored = SessionManager()
        restored.initialize()
        loaded = restored.get_session(session.session_id)

        pd.testing.assert_frame_equal(loaded.df, session.df)
        assert [l.learned for l in loaded.recipe] == [l.learned for l in session.recipe]
        assert loaded.undo().tool == "encode_categorical"  # undo works after a restart
        assert loaded.lock_count == 1


class TestRecipeAPI:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
        with TestClient(app) as c:
            yield c

    def test_list_edit_delete(self, client):
        csv = b"age,city,label\n" + b"\n".join(f"{20 + i if i % 3 else ''},{'ab'[i % 2]},{i % 2}".encode() for i in range(20))
        sid = client.post("/api/upload", files={"file": ("d.csv", csv, "text/csv")}).json()["session_id"]

        def act(tool, **args):
            return client.post("/api/action", json={"session_id": sid, "tool": tool, "arguments": args}).json()

        act("select_target", column="label")
        act("impute_missing", column="age", strategy="mean")
        act("encode_categorical", column="city", method="label")

        recipe = client.get(f"/api/session/{sid}/recipe").json()["recipe"]
        assert [s["tool"] for s in recipe] == ["impute_missing", "encode_categorical"]
        assert recipe[0]["kind"] == "learn" and "fill" in recipe[0]["learned"]

        edited = client.put(f"/api/session/{sid}/recipe/0", json={"args": {"column": "age", "strategy": "median"}})
        assert edited.status_code == 200
        assert edited.json()["new_state_summary"]["recipe"][0]["args"]["strategy"] == "median"

        bad = client.put(f"/api/session/{sid}/recipe/0", json={"args": {"column": "nope"}})
        assert bad.status_code == 400 and "not found" in bad.json()["detail"]

        deleted = client.delete(f"/api/session/{sid}/recipe/0")
        assert [s["tool"] for s in deleted.json()["new_state_summary"]["recipe"]] == ["encode_categorical"]
