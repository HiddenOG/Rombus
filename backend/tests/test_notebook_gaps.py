"""Steps a real Kaggle notebook (Bengaluru house prices) needed that the app couldn't do yet."""

import numpy as np
import pandas as pd
import pytest

from app.expressions import parse_filter
from app.export import pipeline_script
from app.session import DataSession
from app.tools import execute_tool


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


@pytest.fixture
def session():
    """Shaped like Bengaluru_House_Data: text sizes and areas (with ranges), many locations."""
    rng = np.random.default_rng(5)
    n = 2000
    locations = [f"loc{i}" for i in range(60)]
    weights = np.r_[np.full(20, 30), np.full(40, 1)].astype(float)  # 20 common, 40 rare
    bhk = rng.integers(1, 5, n)
    sqft = rng.integers(400, 3000, n)
    area = np.where(rng.random(n) < 0.05, [f"{s} - {s + 400}" for s in sqft], sqft.astype(str))
    price = sqft * rng.normal(0.08, 0.01, n) + bhk * 5
    price[:10] *= 20  # a few absurd prices
    df = pd.DataFrame({
        "location": rng.choice(locations, n, p=weights / weights.sum()),
        "size": [f"{b} BHK" if rng.random() < 0.8 else f"{b} Bedroom" for b in bhk],
        "total_sqft": area,
        "bath": np.minimum(bhk + rng.integers(-1, 4, n), 9).clip(1),
        "price": price.round(2),
    })
    s = DataSession(df, "bengaluru.csv")
    run(s, "select_target", column="price")
    return s


def test_ranges_become_midpoints(session):
    result = run(session, "clean_numeric_string", column="total_sqft")
    assert "became their midpoint" in result["message"]
    raw = session.raw_train["total_sqft"].astype(str)
    i = raw[raw.str.contains("-")].index[0]
    low, high = (float(x) for x in raw[i].split("-"))
    assert session.df.loc[i, "total_sqft"] == (low + high) / 2


def test_plain_numbers_keep_their_type():
    s = DataSession(pd.DataFrame({"a": ["$1,200", "300", None] * 10, "y": [0, 1, 0] * 10}), "d.csv")
    run(s, "select_target", column="y")
    run(s, "clean_numeric_string", column="a")
    assert s.df["a"].dropna().isin([1200, 300]).all()


def test_group_rare_categories_learns_from_training_rows(session):
    result = run(session, "group_rare_categories", column="location", max_rows=10)
    counts = session.raw_train["location"].value_counts()
    common = set(counts[counts > 10].index)
    assert set(session.df["location"].unique()) == common | {"other"}
    assert "values → " in result["message"]
    # Test rows use the training list: a location common only in the test rows still becomes 'other'
    test = session.prepare(session.test_df)
    assert set(test["location"].unique()) <= common | {"other"}


def test_onehot_allows_real_features_but_not_ids(session):
    run(session, "encode_categorical", column="location", method="onehot")  # 60 locations: fine now
    ids = DataSession(pd.DataFrame({"id": [f"x{i}" for i in range(200)], "y": [0, 1] * 100}), "d.csv")
    run(ids, "select_target", column="y")
    assert "looks like an ID" in execute_tool("encode_categorical", ids, {"column": "id", "method": "onehot"})["message"]


def test_remove_outliers_by_group_may_use_the_target(session):
    run(session, "clean_numeric_string", column="total_sqft")
    before = len(session.df)
    result = run(session, "remove_outliers", value="price / total_sqft", group_by="location", method="std", threshold=1)
    # Matches the notebook's per-location rule
    d = session.raw_train.copy()
    from app.recipe import clean_number
    d["pps"] = d["price"] / clean_number(d["total_sqft"])
    g = d.groupby(d["location"].astype(str))["pps"]
    mean, std = g.transform("mean"), g.transform("std")
    expected = int((d["pps"].isna() | std.isna() | d["pps"].between(mean - std, mean + std)).sum())
    assert len(session.df) == expected < before
    assert "aren't dropped" in result["message"]
    assert len(session.prepare(session.test_df)) == len(session.test_df)  # test rows all kept
    assert "pps" not in session.df and "price / total_sqft" not in session.df  # never a feature


def test_remove_outliers_iqr_without_groups(session):
    run(session, "remove_outliers", value="price")
    assert session.df["price"].max() < session.raw_train["price"].max()  # the absurd prices went


@pytest.mark.parametrize("conditions, expected", [
    ("bath > bhk + 2", lambda d: d["bath"] > d["bhk"] + 2),
    ("total_sqft / bhk < 300", lambda d: d["total_sqft"] / d["bhk"] < 300),
    ("bath > bhk + 2 or total_sqft / bhk < 300 and bath > 1",
     lambda d: (d["bath"] > d["bhk"] + 2) | ((d["total_sqft"] / d["bhk"] < 300) & (d["bath"] > 1))),
])
def test_conditions_can_compare_columns(session, conditions, expected):
    run(session, "clean_numeric_string", column="size")
    run(session, "rename_column", column="size", new_name="bhk")
    run(session, "clean_numeric_string", column="total_sqft")
    mask, _ = parse_filter(session.df, conditions)
    assert mask.sum() == int(expected(session.df).fillna(False).sum()) > 0
    run(session, "drop_rows_where", conditions=conditions)
    assert int(expected(session.df).fillna(False).sum()) == 0


def test_linear_regression(session):
    run(session, "clean_numeric_string", column="total_sqft")
    result = run(session, "train_models", algorithms=["linear_regression", "lr"])
    assert set(session.trained_models) == {"linear"}
    assert "Trained 1 model" in result["message"]


def test_the_whole_notebook_the_honest_way(session, tmp_path, monkeypatch):
    """The notebook's cleaning, in the app: and the exported script reproduces it exactly."""
    run(session, "clean_numeric_string", column="size")
    run(session, "rename_column", column="size", new_name="bhk")
    run(session, "clean_numeric_string", column="total_sqft")
    run(session, "impute_missing", column="total_sqft", strategy="median")
    run(session, "add_column", name="bath_per_bhk", formula="bath / bhk")
    run(session, "add_column", name="size_category", formula="(total_sqft > 1000) + (total_sqft > 2000)")
    run(session, "group_rare_categories", column="location", max_rows=10)
    run(session, "drop_rows_where", conditions="total_sqft / bhk < 300 or bath > bhk + 2")
    run(session, "remove_outliers", value="price / total_sqft", group_by="location", threshold=1)
    run(session, "encode_categorical", column="location", method="onehot")
    run(session, "train_models", algorithms=["linear", "rf"])
    exam = run(session, "final_exam")
    assert exam["evidence"]["exam"]["score"] > 0.5

    monkeypatch.chdir(tmp_path)
    session.original_df.to_csv("bengaluru.csv", index=False)
    scope = {"__name__": "pipeline"}
    exec(compile(pipeline_script(session), "pipeline.py", "exec"), scope)
    train, test = scope["main"]("bengaluru.csv")
    pd.testing.assert_frame_equal(train, session.df, check_dtype=False)
    pd.testing.assert_frame_equal(test, session.prepare(session.test_df), check_dtype=False)
