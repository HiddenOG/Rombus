"""Bugs found by acting as users on small projects (Bengaluru by hand, a messy Titanic, stocks over
several files, a customer call list), each pinned down so it stays fixed."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from app.config import settings
from app.session import DataSession
from app.tools import execute_tool
from app.tools.acquire import problem_type_of

H = {"X-Visitor-Id": "user-projects-visitor-0001"}

PEOPLE = pd.DataFrame({
    "Age": [22, 35, None, 41, 29, 52, 33, 47] * 5,
    "Fare £": ["£7.25", "£71.28", "£8.05", "£53.10", "£8.46", "£51.86", "£21.08", "£11.13"] * 5,
    "Survived": ["yes", "no"] * 20,
})


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    with TestClient(app, headers=H) as c:
        yield c


def upload(client, name, data):
    return client.post("/api/upload", files={"file": (name, data)})


# ── Reading files as people save them ────────────────────────────────────────

@pytest.mark.parametrize("name, data", [
    ("semicolons.csv", PEOPLE.to_csv(index=False, sep=";").encode()),  # European Excel
    ("tabs.csv", PEOPLE.to_csv(index=False, sep="\t").encode()),
    ("windows.csv", PEOPLE.to_csv(index=False).encode("cp1252")),  # £ in Windows encoding
    ("bom.csv", PEOPLE.to_csv(index=False).encode("utf-8-sig")),
])
def test_csvs_as_people_save_them(client, name, data):
    r = upload(client, name, data)
    assert r.status_code == 200, r.text
    columns = [c["name"] for c in r.json()["summary"]["columns"]]
    assert columns == ["Age", "Fare £", "Survived"]


def test_an_excel_sheet_with_mixed_columns_uploads(client):
    """A real-world sheet: a title in the header, and columns mixing numbers and text (a heading row,
    totals, notes). Saving it as Parquet crashed the upload, and the crash read as 'cannot connect'."""
    import io as _io
    sheet = pd.DataFrame({
        "SCHEDULE OF RATES FOR PIPELINE MATERIALS": ["S/N", 1, 2, 3, "TOTAL"],
        "Unnamed: 1": ["Description", "Pipe 6in", "Elbow", "Flange", None],
        "Unnamed: 2": ["Qty", 10, 4.5, 2, 16.5],
    })
    buffer = _io.BytesIO()
    sheet.to_excel(buffer, index=False)
    r = upload(client, "rates.xlsx", buffer.getvalue())
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    shown = client.post("/api/action", json={"session_id": sid, "tool": "preview_data", "arguments": {}}).json()
    assert shown["success"]


def test_unexpected_errors_still_reach_the_browser(client, monkeypatch):
    """A crash comes back as a readable 500 with CORS headers, not as 'cannot connect'."""
    from app.routers import upload as upload_router

    def boom(*args):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(upload_router, "_create_notebook", boom)
    with TestClient(app, headers={**H, "Origin": "http://localhost:5173"}, raise_server_exceptions=False) as c:
        r = c.post("/api/upload", files={"file": ("x.csv", b"a,b\n1,2\n")})
    assert r.status_code == 500 and "Something went wrong" in r.json()["detail"]
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "disk on fire" not in r.text  # details stay in the server log


def test_a_binary_file_is_refused_plainly(client):
    r = upload(client, "binary.csv", bytes(range(256)) * 10)
    assert r.status_code == 400 and "isn't a text file" in r.json()["detail"]


# ── The exported script reads the data the same way ──────────────────────────

def run_script(tmp_path, script, files):
    (tmp_path / "pipeline.py").write_text(script, encoding="utf-8")
    for name, data in files:
        (tmp_path / name).write_bytes(data)
    return subprocess.run([sys.executable, "-I", "pipeline.py", *[n for n, _ in files]], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)


def test_script_reads_a_semicolon_file(client, tmp_path):
    data = PEOPLE.to_csv(index=False, sep=";").encode("cp1252")
    sid = upload(client, "people.csv", data).json()["session_id"]
    for tool, args in [("select_target", {"column": "Survived"}), ("clean_numeric_string", {"column": "Fare £"})]:
        assert client.post("/api/action", json={"session_id": sid, "tool": tool, "arguments": args}).json()["success"]
    script = client.get(f"/api/session/{sid}/export", params={"what": "script"}).text
    run = run_script(tmp_path, script, [("people.csv", data)])
    assert run.returncode == 0, run.stderr
    assert pd.read_csv(tmp_path / "train_clean.csv")["Fare £"].dtype.kind == "f"


def test_script_for_several_files_takes_them_all(client, tmp_path):
    files = [(f"{t}.csv", pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d"),
                                        "Close": np.linspace(10, 20, 60) * (i + 1)}).to_csv(index=False).encode())
             for i, t in enumerate(["AAPL", "MSFT"])]
    sid = client.post("/api/upload", files=[("files", f) for f in files]).json()["session_id"]
    r = client.post("/api/action", json={"session_id": sid, "tool": "select_target",
                                         "arguments": {"column": "Close", "split": "time", "time_column": "Date"}})
    assert r.json()["success"]
    script = client.get(f"/api/session/{sid}/export", params={"what": "script"}).text
    assert "AAPL.csv MSFT.csv" in script  # the Run line names the files
    run = run_script(tmp_path, script, files)
    assert run.returncode == 0, run.stderr
    assert "source_file" in pd.read_csv(tmp_path / "train_clean.csv").columns


# ── Settings typed as people (and AIs) type them ─────────────────────────────

@pytest.fixture
def houses():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"size": rng.choice(["2 BHK", "3 BHK", "4 Bedroom"], 120), "sqft": rng.uniform(600, 3000, 120),
                       "society": rng.choice(["a", "b"], 120)})
    df["price"] = df["sqft"] * 0.05 + df["size"].str[0].astype(int) * 10 + rng.normal(0, 5, 120)
    s = DataSession(df, "houses.csv")
    for tool, args in [("select_target", {"column": "price"}), ("remove_columns", {"columns": ["society"]}),
                       ("clean_numeric_string", {"column": "size"}), ("rename_column", {"column": "size", "new_name": "bhk"}),
                       ("train_models", {"algorithms": ["linear"]})]:
        assert execute_tool(tool, s, args)["success"]
    return s


def test_numbers_and_lists_given_as_text(houses):
    assert execute_tool("preview_data", houses, {"rows": "3"})["success"]
    bad = execute_tool("preview_data", houses, {"rows": "many"})
    assert not bad["success"] and bad["message"] == "'rows' must be a number (got 'many')."
    trained = execute_tool("train_models", houses, {"algorithms": "rf, linear"})  # not letter by letter
    assert trained["success"] and set(houses.trained_models) >= {"rf", "linear"}


def test_stand_in_algorithms_are_named(houses):
    message = execute_tool("train_models", houses, {"algorithms": ["xgboost"]})["message"]
    assert "xgboost → gb" in message


def test_predict_takes_a_renamed_column_by_its_new_name(houses):
    by_new_name = execute_tool("predict", houses, {"features": {"bhk": 3, "sqft": 1500}})
    by_original = execute_tool("predict", houses, {"features": {"size": "3 BHK", "sqft": 1500}})
    assert by_new_name["evidence"]["prediction"] == by_original["evidence"]["prediction"]
    # Removed columns aren't asked for; unknown ones are named
    odd = execute_tool("predict", houses, {"features": {"bhk": 3, "sqft": 1500, "colour": "red"}})["message"]
    assert "society" not in odd and "Ignored (not columns of the original data): colour" in odd


# ── Messages that say what's actually wrong ──────────────────────────────────

def test_a_missing_column_in_a_formula_condition_is_named(houses):
    r = execute_tool("drop_rows_where", houses, {"conditions": "sqft / bedrooms < 300"})
    assert r["message"] == "Column 'bedrooms' not found."


def test_classes_seen_once_get_a_plain_message():
    s = DataSession(pd.DataFrame({"x": range(30), "y": ["a"] * 25 + ["b", "c", "d", "e", "f"]}), "t.csv")
    execute_tool("select_target", s, {"column": "y", "test_size": 0.2})
    r = execute_tool("train_models", s, {"algorithms": ["rf"]})
    assert not r["success"] and "appear only once" in r["message"]


@pytest.mark.parametrize("values, kind", [
    (list(range(10)), "regression"),            # 10 prices in a 10-row file are amounts
    ([0, 1] * 20, "classification"),
    ([1, 2, 3, 4, 5] * 8, "classification"),     # ratings
    ([1.5, 2.25, 3.0] * 10, "regression"),       # decimals are amounts
    (["yes", "no"] * 5, "classification"),
    ([True, False] * 5, "classification"),
])
def test_problem_type(values, kind):
    assert problem_type_of(pd.Series(values)) == kind
