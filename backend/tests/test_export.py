"""Exports: cleaned data files and a pipeline script that reproduces the app exactly."""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from app.config import settings
from app.export import cleaned_rows, pipeline_script
from app.session import DataSession
from app.tools import execute_tool

VISITOR = {"X-Visitor-Id": "export-visitor-00000001"}

HONEST_CUSTOM = '''
def learn(train):
    return {"cap": float(train["age"].quantile(0.95))}

def apply(df, learned):
    df["age"] = df["age"].clip(upper=learned["cap"])
    return df
'''


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]


@pytest.fixture
def session():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "customer_id": range(n),
        "age": np.where(rng.random(n) < 0.15, np.nan, rng.integers(18, 80, n)).astype(float),
        "income": [f"${v:,}" for v in rng.integers(20, 120, n) * 1000],
        "city": rng.choice(["London", "Paris", "Berlin"], n),
        "churned": rng.choice(["yes", "no"], n),
    })
    df.loc[5] = df.loc[4]  # a duplicate row for the train-only step
    s = DataSession(df, "customers.csv")
    run(s, "select_target", column="churned", test_size=0.25)
    run(s, "remove_columns", columns=["customer_id"])
    run(s, "remove_duplicates")
    run(s, "clean_numeric_string", column="income")
    run(s, "impute_all_missing")
    run(s, "custom_code", code=HONEST_CUSTOM)
    run(s, "encode_all_categorical", method="onehot")
    run(s, "scale_all_features")
    return s


def test_all_rows_uses_training_notes_for_test_rows(session):
    everything = cleaned_rows(session, "all")
    assert set(everything["split"]) == {"train", "test"}
    assert (everything["split"] == "test").sum() == len(session.test_df)
    pd.testing.assert_frame_equal(
        everything[everything["split"] == "test"].drop(columns="split").reset_index(drop=True),
        session.prepare(session.test_df), check_dtype=False)


def test_pipeline_script_reproduces_the_app_exactly(session, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session.original_df.to_csv("customers.csv", index=False)
    scope = {"__name__": "pipeline"}
    exec(compile(pipeline_script(session), "pipeline.py", "exec"), scope)
    train_clean, test_clean = scope["main"]("customers.csv")

    pd.testing.assert_frame_equal(train_clean, session.df, check_dtype=False, atol=1e-6)
    pd.testing.assert_frame_equal(test_clean, session.prepare(session.test_df), check_dtype=False, atol=1e-6)
    assert (tmp_path / "train_clean.csv").exists() and (tmp_path / "test_clean.csv").exists()


def test_download_endpoints(session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    with TestClient(app, headers=VISITOR) as client:
        csv = b"age,city,churned\n" + b"\n".join(f"{20 + i},{'ab'[i % 2]},{'yn'[i % 2]}".encode() for i in range(30))
        sid = client.post("/api/upload", files={"file": ("people data.csv", csv, "text/csv")}).json()["session_id"]
        client.post("/api/action", json={"session_id": sid, "tool": "select_target", "arguments": {"column": "churned"}})

        r = client.get(f"/api/session/{sid}/export", params={"what": "train", "format": "csv"})
        assert r.status_code == 200 and 'filename="people_data_train_clean.csv"' in r.headers["content-disposition"]
        assert len(pd.read_csv(io.BytesIO(r.content))) == 24

        r = client.get(f"/api/session/{sid}/export", params={"what": "all", "format": "xlsx"})
        assert len(pd.read_excel(io.BytesIO(r.content))) == 30

        r = client.get(f"/api/session/{sid}/export", params={"what": "script"})
        assert r.status_code == 200 and b"def fit_recipe" in r.content

        assert client.get(f"/api/session/{sid}/export", params={"what": "train"},
                          headers={"X-Visitor-Id": "someone-else-000000001"}).status_code == 404
