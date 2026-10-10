"""Jupyter-style controls: restart, move steps, and the AI knowing that nothing is final."""

import numpy as np
import pandas as pd
import pytest

from app.copilot_prompt import STATIC_INSTRUCTIONS, state_snapshot
from app.session import DataSession
from app.tools import execute_tool


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


@pytest.fixture
def session():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": np.where(rng.random(80) < 0.2, np.nan, rng.integers(18, 70, 80)),
        "income": [f"${v:,}" for v in rng.integers(20, 90, 80) * 1000],
        "churned": rng.choice(["yes", "no"], 80),
    })
    s = DataSession(df, "d.csv")
    run(s, "select_target", column="churned")
    return s


def test_restart_clears_models_and_ai_memory_but_keeps_the_recipe(session):
    run(session, "impute_all_missing")
    run(session, "clean_numeric_string", column="income")
    run(session, "train_models", algorithms=["lr"], cv_folds=2)
    session.memory = [{"role": "user", "content": "hi"}]
    before = session.df.copy()

    session.restart()

    assert session.trained_models == {} and session.X_train is None
    assert session.memory == []
    assert len(session.recipe) == 2
    pd.testing.assert_frame_equal(session.df, before)  # rebuilt from scratch, identical


def test_move_step_reorders_and_replays(session):
    first = session.add_step("impute_all_missing", {})
    second = session.add_step("clean_numeric_string", {"column": "income"})
    session.move_step(second.id, -1)
    assert [l.id for l in session.recipe] == [second.id, first.id]


def test_move_that_breaks_a_step_is_refused(session):
    clean = session.add_step("clean_numeric_string", {"column": "income"})
    log = session.add_step("log_transform", {"column": "income"})  # needs numbers first
    with pytest.raises(ValueError, match="break a later step"):
        session.move_step(log.id, -1)
    assert [l.id for l in session.recipe] == [clean.id, log.id]


def test_move_past_the_end(session):
    only = session.add_step("impute_all_missing", {})
    with pytest.raises(ValueError, match="already at the start"):
        session.move_step(only.id, -1)


def test_ai_is_told_nothing_is_final(session):
    assert "Nothing in the notebook is final" in STATIC_INSTRUCTIONS
    snapshot = state_snapshot(session)
    assert "snapshots" not in snapshot  # the old wording that misled the AI
    assert "LOCKED 1 TIME" in snapshot
