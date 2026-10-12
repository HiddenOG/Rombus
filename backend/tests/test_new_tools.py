"""Tools added after watching real use: drop rows with blanks, list unique values, map the blanks."""

import numpy as np
import pandas as pd
import pytest

from app.export import pipeline_script
from app.session import DataSession
from app.tools import execute_tool


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


@pytest.fixture
def session():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "size": np.where(rng.random(n) < 0.1, None, rng.choice(["2 BHK", "3 BHK", "1 BHK"], n)),
        "bath": np.where(rng.random(n) < 0.05, np.nan, rng.integers(1, 4, n)),
        "area": rng.integers(500, 3000, n),
        "price": rng.integers(20, 200, n),
    })
    s = DataSession(df, "houses.csv")
    run(s, "select_target", column="price")
    return s


def test_drop_missing_rows_only_touches_training_rows(session):
    before = len(session.df)
    blanks = int(session.df["size"].isna().sum())
    result = run(session, "drop_missing_rows", columns=["size"])
    assert len(session.df) == before - blanks and session.df["size"].notna().all()
    assert "Test rows and new customers aren't dropped" in result["message"]
    # The locked test rows keep every row, blanks included
    prepared = session.prepare(session.test_df)
    assert len(prepared) == len(session.test_df)


def test_drop_missing_rows_without_columns_means_any_column(session):
    run(session, "drop_missing_rows")
    assert not session.df.isna().any().any()


def test_drop_missing_rows_refuses_unknown_columns(session):
    assert "not found" in execute_tool("drop_missing_rows", session, {"columns": ["nope"]})["message"]


def test_exported_script_matches_with_the_new_step(session, tmp_path, monkeypatch):
    run(session, "drop_missing_rows", columns=["size"])
    run(session, "impute_all_missing")
    run(session, "encode_all_categorical", method="label")
    monkeypatch.chdir(tmp_path)
    session.original_df.to_csv("houses.csv", index=False)
    scope = {"__name__": "pipeline"}
    exec(compile(pipeline_script(session), "pipeline.py", "exec"), scope)
    train, test = scope["main"]("houses.csv")
    pd.testing.assert_frame_equal(train, session.df, check_dtype=False)
    pd.testing.assert_frame_equal(test, session.prepare(session.test_df), check_dtype=False)


def test_show_unique_values(session):
    result = run(session, "show_unique_values", column="size")
    assert "'2 BHK'" in result["message"] and "None" in result["message"]
    table = result["evidence"]["preview"]
    assert sum(r["rows"] for r in table) == len(session.df)


def test_missing_map(session):
    result = run(session, "plot_missing")
    ev = result["evidence"]
    assert ev["chart_type"] == "missing_map" and ev["columns"][0] == "size"  # most blanks first
    assert ev["missing_share"]["area"] == 0.0
    assert len(ev["rows"]) <= 120 and len(ev["rows"][0]) == len(ev["columns"])
    assert "have blanks" in result["message"]
