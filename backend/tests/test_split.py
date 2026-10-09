"""Tests for locking away the test rows when the target is chosen (strict mode)."""

import pandas as pd
import pytest

from app.gemini import build_tool_declarations
from app.session import DataSession, SessionManager
from app.tools import BEFORE_SPLIT_TOOLS, execute_tool
from app.tools.acquire import parse_test_size


@pytest.fixture
def session():
    df = pd.DataFrame({
        "row_id": range(100),
        "age": [20 + i % 40 for i in range(100)],
        "churned": ["yes"] * 30 + ["no"] * 70,
    })
    return DataSession(df, "customers.csv")


def lock(session, **args):
    result = execute_tool("select_target", session, {"column": "churned", **args})
    assert result["success"], result["message"]
    return result


class TestBeforeLock:
    def test_statistics_hidden(self, session):
        col = session.get_summary()["columns"][1]
        assert col == {"name": "age", "dtype": "int64"}

    def test_cleaning_blocked(self, session):
        result = execute_tool("impute_all_missing", session, {})
        assert not result["success"]
        assert "target column first" in result["message"]

    def test_charts_blocked(self, session):
        assert not execute_tool("plot_histogram", session, {"column": "age"})["success"]

    def test_preview_capped(self, session):
        result = execute_tool("preview_data", session, {"rows": 50})
        assert len(result["evidence"]["preview"]) == 10

    def test_ai_only_offered_allowed_tools(self, session):
        names = {d["name"] for d in build_tool_declarations(session)}
        assert names == BEFORE_SPLIT_TOOLS


class TestLocking:
    def test_default_twenty_percent(self, session):
        lock(session)
        assert len(session.test_df) == 20
        assert len(session.df) == 80

    @pytest.mark.parametrize("value", [0.3, 30])
    def test_choose_thirty_percent(self, session, value):
        result = lock(session, test_size=value)
        assert len(session.test_df) == 30
        assert "30%" in result["message"]

    def test_ai_sending_null_test_size_uses_default(self, session):
        # Gemini sometimes fills optional arguments with null instead of leaving them out
        lock(session, test_size=None)
        assert len(session.test_df) == 20

    def test_null_arguments_fall_back_to_defaults_everywhere(self, session):
        result = execute_tool("preview_data", session, {"rows": None})
        assert result["success"], result["message"]
        assert len(result["evidence"]["preview"]) == 5

    def test_no_row_in_both_piles(self, session):
        lock(session)
        assert set(session.df["row_id"]).isdisjoint(session.test_df["row_id"])
        assert len(session.df) + len(session.test_df) == 100

    def test_same_mix_of_answers_in_both_piles(self, session):
        lock(session)
        assert (session.test_df["churned"] == "yes").mean() == pytest.approx(0.3)
        assert (session.df["churned"] == "yes").mean() == pytest.approx(0.3)

    def test_blank_answers_removed(self, session):
        session.original_df.loc[:4, "churned"] = None
        result = lock(session)
        assert "Removed 5 rows" in result["message"]
        assert len(session.df) + len(session.test_df) == 95

    def test_statistics_come_from_training_rows_only(self, session):
        lock(session)
        col = next(c for c in session.get_summary()["columns"] if c["name"] == "age")
        assert col["mean"] == pytest.approx(session.df["age"].mean(), abs=1e-4)

    def test_everything_unlocked_after(self, session):
        lock(session)
        assert execute_tool("plot_histogram", session, {"column": "age"})["success"]


class TestChangingYourMind:
    def test_can_relock_before_cleaning(self, session):
        lock(session, test_size=0.2)
        lock(session, test_size=0.3)
        assert len(session.test_df) == 30
        assert len(session.df) == 70

    def test_cannot_relock_after_cleaning(self, session):
        lock(session)
        execute_tool("remove_columns", session, {"columns": ["row_id"]})
        result = execute_tool("select_target", session, {"column": "churned", "test_size": 0.3})
        assert not result["success"]
        assert "Undo" in result["message"]


class TestTestSizeLimits:
    @pytest.mark.parametrize("value,expected", [(0.2, 0.2), (25, 0.25), (5, 0.05), (0.5, 0.5)])
    def test_valid(self, value, expected):
        assert parse_test_size(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [0.01, 0.7, 70, 1])
    def test_out_of_range(self, value):
        with pytest.raises(ValueError, match="between 5% and 50%"):
            parse_test_size(value)


def test_lock_survives_restart(session, tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    manager = SessionManager()
    manager.initialize()
    lock(session, test_size=0.3)
    manager.save(session)

    reloaded = SessionManager()
    reloaded.initialize()
    restored = reloaded.get_session(session.session_id)
    assert restored.is_split
    assert restored.test_size == pytest.approx(0.3)
    assert restored.test_df["row_id"].tolist() == session.test_df["row_id"].tolist()
