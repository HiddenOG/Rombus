"""Analysis views: filter rows, group & summarise, pivot table, and preview first/last/random."""

import numpy as np
import pandas as pd
import pytest

from app.action_code import code_info
from app.session import DataSession
from app.tools import execute_tool


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


@pytest.fixture
def session():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "location": rng.choice(["Whitefield", "Hebbal", "Sarjapur Road"], n),
        "size": np.where(rng.random(n) < 0.1, None, rng.choice(["2 BHK", "3 BHK"], n)),
        "total sqft": rng.integers(500, 3000, n).astype(float),
        "bath": rng.integers(1, 4, n),
        "price": rng.integers(20, 300, n),
    })
    s = DataSession(df, "houses.csv")
    run(s, "select_target", column="price")
    return s


@pytest.mark.parametrize("conditions, expected", [
    ("price > 100", lambda d: d["price"] > 100),
    ("price >= 100 and bath == 2", lambda d: (d["price"] >= 100) & (d["bath"] == 2)),
    ('location == "Whitefield" or bath < 2', lambda d: (d["location"] == "Whitefield") | (d["bath"] < 2)),
    ("location contains sarjapur", lambda d: d["location"].str.contains("Sarjapur")),
    ("size is blank", lambda d: d["size"].isna()),
    ("size is not blank and total sqft > 1500", lambda d: d["size"].notna() & (d["total sqft"] > 1500)),  # spaces in names
    ("`total sqft` <= 900", lambda d: d["total sqft"] <= 900),
])
def test_filter_matches_pandas(session, conditions, expected):
    result = run(session, "filter_rows", conditions=conditions)
    want = int(expected(session.df).sum())
    assert result["message"].startswith(f"{want} of {len(session.df)} training rows")


@pytest.mark.parametrize("conditions", [
    "__import__('os').system('echo hi')",     # code is never run: it's not even a column
    "price > abc",                            # not a number
    "location > 5",                           # can't order text
    "price",                                  # no operator
    "",
])
def test_filter_refuses_what_it_cant_read(session, conditions):
    result = execute_tool("filter_rows", session, {"conditions": conditions})
    assert not result["success"] and "Something went wrong" not in result["message"]


def test_filter_never_changes_the_data(session):
    before = session.df.copy()
    run(session, "filter_rows", conditions="price > 250")
    pd.testing.assert_frame_equal(session.df, before)
    assert not session.recipe


def test_group_summary(session):
    result = run(session, "group_summary", group_by="location", value="price", aggregation="mean")
    expected = session.df.groupby("location")["price"].mean().sort_values(ascending=False)
    table = result["evidence"]["preview"]
    assert [r["location"] for r in table] == list(expected.index)
    assert table[0]["mean of price"] == pytest.approx(expected.iloc[0], abs=1e-3)
    assert result["evidence"]["chart_type"] == "bar"


def test_group_summary_counts_and_blank_groups(session):
    table = run(session, "group_summary", group_by="size")["evidence"]["preview"]
    assert {r["size"] for r in table} == {"2 BHK", "3 BHK", "(blank)"}
    assert sum(r["rows"] for r in table) == len(session.df)


def test_group_summary_refuses_mean_of_text(session):
    assert "holds text" in execute_tool("group_summary", session, {"group_by": "bath", "value": "location"})["message"]


def test_pivot_table(session):
    result = run(session, "pivot_table", rows="location", columns="bath", value="price", aggregation="median")
    table = pd.DataFrame(result["evidence"]["preview"]).set_index("location")
    expected = session.df.pivot_table(index="location", columns="bath", values="price", aggfunc="median")
    assert table.loc["Hebbal", "2"] == pytest.approx(expected.loc["Hebbal", 2])


def test_pivot_counts_without_a_value(session):
    table = pd.DataFrame(run(session, "pivot_table", rows="location", columns="bath")["evidence"]["preview"])
    assert table.drop(columns="location").to_numpy().sum() == len(session.df)


@pytest.mark.parametrize("position, rows", [("first", [0, 1]), ("last", [-2, -1])])
def test_preview_first_and_last(session, position, rows):
    shown = run(session, "preview_data", rows=2, position=position)["evidence"]["preview"]
    assert [r["price"] for r in shown] == [int(session.df["price"].iloc[i]) for i in rows]


def test_preview_random_and_capped(session):
    assert len(run(session, "preview_data", rows=10, position="random")["evidence"]["preview"]) == 10
    assert len(run(session, "preview_data", rows=100_000)["evidence"]["preview"]) == 200


def test_filter_code_tab_is_plain_pandas(session):
    code = code_info(session, "filter_rows", {"conditions": 'price > 100 and location == "Hebbal"'})["code"]
    assert "train['price'] > 100.0" in code and "train['location'] == 'Hebbal'" in code
