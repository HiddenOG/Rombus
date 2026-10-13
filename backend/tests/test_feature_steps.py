"""Cleaning and feature steps: remove rows where, change type, rename, fill a value, new column from a formula."""

import numpy as np
import pandas as pd
import pytest

from app.expressions import evaluate_formula
from app.export import pipeline_script
from app.session import DataSession
from app.tools import execute_tool


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


def refused(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert not result["success"], f"{tool} should have been refused"
    assert "Something went wrong" not in result["message"]
    return result["message"]


@pytest.fixture
def session():
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame({
        "QUANTITY": rng.integers(1, 50, n),
        "PRICE": np.where(rng.random(n) < 0.03, 99999, rng.integers(10, 200, n)),  # a few outliers
        "DISCOUNT": np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 20, n)),
        "rooms text": rng.choice(["1", "2", "3", "three"], n),
        "Origin": np.where(rng.random(n) < 0.1, None, rng.choice(["BH", "WB", "MP"], n)),
        "sold": rng.choice(["yes", "no"], n),
    })
    s = DataSession(df, "fruits.csv")
    run(s, "select_target", column="sold")
    return s


def test_drop_rows_where_only_touches_training_rows(session):
    outliers = int((session.df["PRICE"] > 10000).sum())
    result = run(session, "drop_rows_where", conditions="PRICE > 10000")
    assert outliers > 0 and (session.df["PRICE"] <= 10000).all()
    assert "keep theirs" in result["message"]
    assert len(session.prepare(session.test_df)) == len(session.test_df)


def test_change_type(session):
    result = run(session, "change_type", column="rooms text", to="whole number")
    assert str(session.df["rooms text"].dtype) == "Int64"
    assert "couldn't be converted" in result["message"]  # "three"
    run(session, "change_type", column="QUANTITY", to="text")
    assert pd.api.types.is_string_dtype(session.df["QUANTITY"]) or session.df["QUANTITY"].dtype == object
    assert "target" in refused(session, "change_type", column="sold", to="text")


def test_rename_column(session):
    run(session, "rename_column", column="DISCOUNT", new_name="discount_pct")
    assert "discount_pct" in session.df and "DISCOUNT" not in session.df
    assert "already" in refused(session, "rename_column", column="PRICE", new_name="QUANTITY")
    assert "target" in refused(session, "rename_column", column="sold", new_name="label")


def test_fill_value(session):
    run(session, "fill_value", column="Origin", value="unknown")
    assert (session.df["Origin"] == "unknown").sum() > 0 and session.df["Origin"].notna().all()
    run(session, "fill_value", column="DISCOUNT", value="0")
    assert session.df["DISCOUNT"].notna().all()
    assert "number" in refused(session, "fill_value", column="PRICE", value="lots")


def test_add_column_formula(session):
    run(session, "add_column", name="Paid", formula="QUANTITY * PRICE - QUANTITY * PRICE * DISCOUNT / 100")
    d = session.df
    expected = d["QUANTITY"] * d["PRICE"] - d["QUANTITY"] * d["PRICE"] * d["DISCOUNT"] / 100
    pd.testing.assert_series_equal(d["Paid"], expected.astype(float), check_names=False)


@pytest.mark.parametrize("formula, expected", [
    ("log(PRICE)", lambda d: np.log(d["PRICE"])),
    ("PRICE > 100", lambda d: (d["PRICE"] > 100).astype(float)),
    ("-QUANTITY ** 2", lambda d: -(d["QUANTITY"].astype(float) ** 2)),
    ("PRICE / (QUANTITY - QUANTITY)", lambda d: pd.Series(np.nan, index=d.index)),  # divide by zero -> blank
])
def test_formula_values(session, formula, expected):
    got = evaluate_formula(session.df, formula)
    pd.testing.assert_series_equal(got, expected(session.df).astype(float), check_names=False)


@pytest.mark.parametrize("formula", [
    "__import__('os').system('echo hacked')",
    "PRICE.__class__",
    "[x for x in range(10**9)]",
    "open('secrets.txt')",
    "eval('1+1')",
    "PRICE if True else 0",
    "'text' * 3",
    "lambda: 1",
    "Origin * 2",               # a text column
    "sold * 2",                 # the target: answers can't become features
    "no_such_column + 1",
    "",
])
def test_formulas_refuse_anything_but_arithmetic(session, formula):
    refused(session, "add_column", name="x", formula=formula)
    assert "x" not in session.df


def test_huge_powers_dont_hang(session):
    result = execute_tool("add_column", session, {"name": "big", "formula": "10 ** 10 ** 10 + PRICE"})
    # Overflow -> blank, instantly; a column with nothing in it is refused rather than added
    assert not result["success"] and "no usable value" in result["message"] and "big" not in session.df


def test_backticks_for_names_with_spaces(session):
    run(session, "change_type", column="rooms text", to="number")
    run(session, "add_column", name="per_room", formula="PRICE / `rooms text`")
    assert "per_room" in session.df


def test_exported_script_reproduces_all_five(session, tmp_path, monkeypatch):
    run(session, "drop_rows_where", conditions="PRICE > 10000 or QUANTITY < 2")
    run(session, "change_type", column="rooms text", to="whole number")  # its blanks get a rounded mean below
    run(session, "rename_column", column="DISCOUNT", new_name="discount")
    run(session, "fill_value", column="discount", value="0")
    run(session, "fill_value", column="Origin", value="unknown")
    run(session, "add_column", name="paid", formula="QUANTITY * PRICE * (1 - discount / 100)")
    run(session, "impute_all_missing")
    run(session, "encode_all_categorical", method="onehot")

    monkeypatch.chdir(tmp_path)
    session.original_df.to_csv("fruits.csv", index=False)
    scope = {"__name__": "pipeline"}
    exec(compile(pipeline_script(session), "pipeline.py", "exec"), scope)
    train, test = scope["main"]("fruits.csv")
    pd.testing.assert_frame_equal(train, session.df, check_dtype=False)
    pd.testing.assert_frame_equal(test, session.prepare(session.test_df), check_dtype=False)


def test_steps_train_and_predict_end_to_end(session):
    run(session, "change_type", column="rooms text", to="whole number")
    run(session, "add_column", name="paid", formula="QUANTITY * PRICE")
    run(session, "impute_all_missing")
    run(session, "encode_all_categorical", method="label")
    run(session, "train_models", algorithms=["rf", "lr"])
    run(session, "final_exam")
    raw = {"QUANTITY": 5, "PRICE": 50, "DISCOUNT": None, "rooms text": "2", "Origin": "BH"}
    assert run(session, "predict", features=raw)["evidence"]["prediction"] in ("yes", "no")
