"""Time series, from a stock-analysis notebook: several files stacked, dates, line charts, returns and
moving averages that never look ahead, and a test lock on the MOST RECENT rows."""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from app.config import settings
from app.export import pipeline_script
from app.expressions import parse_filter
from app.session import DataSession
from app.tools import execute_tool

H = {"X-Visitor-Id": "time-series-visitor-0001"}


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


def prices(ticker, seed, days=120):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, days))
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": close.round(2),
                         "Volume": rng.integers(1_000, 9_000, days)})


@pytest.fixture
def stocks():
    """Three tickers stacked, like uploading AAPL.csv, MSFT.csv and NVDA.csv together."""
    frames = [prices(t, i).assign(source_file=t) for i, t in enumerate(["AAPL", "MSFT", "NVDA"])]
    return pd.concat(frames, ignore_index=True)


def test_uploading_several_files_stacks_them(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    with TestClient(app, headers=H) as client:
        files = [("files", (f"{t}.csv", prices(t, i).to_csv(index=False).encode())) for i, t in enumerate(["AAPL", "MSFT"])]
        r = client.post("/api/upload", files=files)
        assert r.status_code == 200 and r.json()["files"] == 2
        summary = r.json()["summary"]
        assert summary["shape"] == [240, 4] and summary["filename"] == "AAPL + 1 more.csv"


def test_time_split_locks_the_most_recent_rows(stocks):
    s = DataSession(stocks, "stocks.csv")
    result = run(s, "select_target", column="Close", split="time", time_column="Date")
    train_last = pd.to_datetime(s.raw_train["Date"]).max()
    test_first = pd.to_datetime(s.test_df["Date"]).min()
    assert train_last <= test_first and len(s.test_df) == 72  # 20% of 360
    assert "always tested on a future" in result["message"]


def test_time_split_refuses_non_dates(stocks):
    s = DataSession(stocks, "stocks.csv")
    assert "doesn't look like dates" in execute_tool(
        "select_target", s, {"column": "Close", "split": "time", "time_column": "source_file"})["message"]


def test_date_parts(stocks):
    s = DataSession(stocks, "stocks.csv")
    run(s, "clean_only")
    run(s, "date_parts", column="Date", parts=["year", "month", "weekday", "weekday name"])
    assert set(s.df["Date_weekday"]) <= {0, 1, 2, 3, 4}  # business days
    assert s.df.loc[0, "Date_weekday_name"] == "Monday" and s.df.loc[0, "Date_year"] == 2023


def test_date_range_conditions(stocks):
    s = DataSession(stocks, "stocks.csv")
    run(s, "clean_only")
    mask, _ = parse_filter(s.df, "Date >= 2023-03-01 and Date < 2023-04-01")
    assert mask.sum() == 3 * 23  # March 2023 has 23 business days, for each of 3 tickers
    run(s, "filter_rows", conditions="Date >= 2023-03-01")


def test_returns_and_moving_average_per_company(stocks):
    s = DataSession(stocks, "stocks.csv")
    run(s, "clean_only")
    run(s, "time_feature", column="Close", kind="change %", order_by="Date", group_by="source_file")
    run(s, "time_feature", column="Close", kind="moving average", window=10, order_by="Date", group_by="source_file")
    for t, rows in s.df.groupby("source_file"):
        rows = rows.sort_values("Date")
        expected_ret = rows["Close"].pct_change() * 100
        expected_ma = rows["Close"].rolling(10).mean()
        pd.testing.assert_series_equal(rows["Close_change_pct_1"], expected_ret, check_names=False)
        pd.testing.assert_series_equal(rows["Close_ma_10"], expected_ma, check_names=False)


def test_test_rows_get_history_and_nothing_looks_ahead(stocks):
    s = DataSession(stocks, "stocks.csv")
    run(s, "select_target", column="Volume", split="time", time_column="Date")
    run(s, "time_feature", column="Close", kind="moving average", window=5, group_by="source_file")
    test = s.prepare(s.test_df)
    # The first test day of each company already has its 5-day average: history from the training rows
    assert test["Close_ma_5"].notna().all()
    # ... and it equals the average over the whole series computed in time order (past only, no peeking)
    full = stocks.sort_values(["source_file", "Date"]).copy()
    full["ma"] = full.groupby("source_file")["Close"].transform(lambda v: v.rolling(5).mean())
    merged = test.merge(full[["Date", "source_file", "ma"]], on=["Date", "source_file"])
    np.testing.assert_allclose(merged["Close_ma_5"], merged["ma"])


def test_the_target_only_uses_earlier_rows(stocks):
    s = DataSession(stocks, "stocks.csv")
    run(s, "select_target", column="Close", split="time", time_column="Date")
    result = run(s, "time_feature", column="Close", kind="moving average", window=3, group_by="source_file")
    assert "uses only EARLIER rows" in result["message"]
    rows = s.df[s.df["source_file"] == "AAPL"].sort_values("Date")
    expected = rows["Close"].shift(1).rolling(3).mean()  # yesterday and before, never today's answer
    pd.testing.assert_series_equal(rows["Close_ma_3"], expected, check_names=False)


def test_line_chart(stocks):
    s = DataSession(stocks, "stocks.csv")
    run(s, "clean_only")
    ev = run(s, "plot_line", x="Date", y=["Close"], group_by="source_file")["evidence"]
    assert ev["chart_type"] == "line" and ev["x_is_date"] and [l["name"] for l in ev["series"]] == ["AAPL", "MSFT", "NVDA"]
    first = ev["series"][0]["points"][0]
    assert pd.Timestamp(first[0], unit="ms") == pd.Timestamp("2023-01-02")
    ev = run(s, "plot_line", x="Date", y=["Close", "Volume"])["evidence"]
    assert [l["name"] for l in ev["series"]] == ["Close", "Volume"]


def test_correlation_between_companies_and_covariance(stocks):
    s = DataSession(stocks, "stocks.csv")
    run(s, "clean_only")
    run(s, "time_feature", column="Close", kind="change %", order_by="Date", group_by="source_file")
    ev = run(s, "plot_correlation", value="Close_change_pct_1", group_by="source_file", align_on="Date")["evidence"]
    assert ev["columns"] == ["AAPL", "MSFT", "NVDA"] and ev["correlation"]["AAPL"]["AAPL"] == 1.0
    cov = run(s, "plot_correlation", measure="covariance")["evidence"]["preview"]
    assert {"Close", "Volume"} <= set(cov[0])


def test_group_summary_of_several_columns(stocks):
    s = DataSession(stocks, "stocks.csv")
    run(s, "clean_only")
    table = run(s, "group_summary", group_by="source_file", value="Close, Volume", aggregation="mean")["evidence"]["preview"]
    assert len(table) == 3 and {"mean of Close", "mean of Volume"} <= set(table[0])


def test_dates_never_become_one_hot_columns(stocks):
    s = DataSession(stocks, "stocks.csv")
    run(s, "select_target", column="Close", split="time", time_column="Date")
    result = run(s, "train_models", algorithms=["rf"])
    assert "Date" not in "".join(s.X_train.columns) and "Not used as model inputs: ['Date']" in result["message"]
    assert "the check rows are the latest ones" in result["message"]
    assert all(isinstance(c, str) for c in s.X_train.columns)


def test_the_whole_thing_exports_exactly(stocks, tmp_path, monkeypatch):
    s = DataSession(stocks, "stocks.csv")
    run(s, "select_target", column="Close", split="time", time_column="Date")
    run(s, "time_feature", column="Close", kind="change %", group_by="source_file")
    run(s, "time_feature", column="Volume", kind="moving average", window=5, group_by="source_file")
    run(s, "date_parts", column="Date", parts=["month", "weekday"])
    run(s, "encode_categorical", column="source_file", method="onehot")
    run(s, "train_models", algorithms=["linear", "rf"])
    run(s, "final_exam")

    monkeypatch.chdir(tmp_path)
    s.original_df.to_csv("stocks.csv", index=False)
    scope = {"__name__": "pipeline"}
    exec(compile(pipeline_script(s), "pipeline.py", "exec"), scope)
    train, test = scope["main"]("stocks.csv")
    pd.testing.assert_frame_equal(train, s.df, check_dtype=False)
    pd.testing.assert_frame_equal(test, s.prepare(s.test_df), check_dtype=False)
