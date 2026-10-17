"""The wider model library: every traditional regression and classification model, a neural network,
clustering as a step, and Models cells with their own cross-validation and tuning."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from app import sandbox
from app.action_code import code_info
from app.config import settings
from app.export import pipeline_script
from app.session import DataSession
from app.tools import execute_tool
from app.tools.model import CLASSIFICATION_MODELS, REGRESSION_MODELS, resolve_algorithms


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


def notebook(kind="regression", n=400, time=False):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 5, n).astype(float)})
    df["y"] = (3 * df.a - 2 * df.b ** 2 + df.c + rng.normal(0, 0.5, n) if kind == "regression"
               else ((df.a + df.b + rng.normal(0, 0.5, n)) > 0).astype(int))
    if time:
        df["date"] = pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    s = DataSession(df, "t.csv")
    run(s, "select_target", column="y", **({"split": "time", "time_column": "date"} if time else {}))
    if time:
        run(s, "remove_columns", columns=["date"])
    return s


@pytest.mark.parametrize("kind, registry", [("regression", REGRESSION_MODELS), ("classification", CLASSIFICATION_MODELS)])
def test_every_built_in_model_trains_and_its_code_runs(kind, registry):
    s = notebook(kind)
    results = run(s, "train_models", algorithms=list(registry))["evidence"]["results"]
    assert [r["model"] for r in results] == list(registry)
    assert all("cv_mean" in r for r in results), [r for r in results if "error" in r]
    compile(code_info(s, "train_models", {"algorithms": list(registry)})["code"], "<code tab>", "exec")


def test_everyday_names_find_the_right_model():
    assert resolve_algorithms(["Neural Network", "extra trees", "Naive Bayes", "SVM RBF"], "classification")[0] == \
        ["mlp", "et", "nb", "svc_rbf"]
    assert resolve_algorithms(["svm", "logistic regression", "polynomial regression"], "regression")[0] == \
        ["svr", "linear", "poly"]


def test_scaled_models_keep_the_scaler_inside():
    """A neural network or SVM learns its scaling within each training fold: no leak into the checks."""
    model = REGRESSION_MODELS["mlp"]()
    assert type(model).__name__ == "Pipeline" and type(model.steps[0][1]).__name__ == "StandardScaler"
    s = notebook()
    assert "make_pipeline(StandardScaler(), MLPRegressor(" in code_info(s, "train_models", {"algorithms": ["mlp"]})["code"]


def houses_with_areas(n=600):
    """Price depends strongly on the area: a text column with many values, which one-hot leaves out."""
    rng = np.random.default_rng(3)
    areas = [f"area_{i}" for i in range(150)]
    premium = {a: rng.uniform(50, 200) for a in areas}
    df = pd.DataFrame({"location": rng.choice(areas, n), "sqft": rng.uniform(500, 3000, n),
                       "Unnamed: 1": rng.normal(size=n)})  # a name LightGBM can't take as it is
    df["price"] = df.sqft * 0.05 + df.location.map(premium) + rng.normal(0, 5, n)
    s = DataSession(df, "h.csv")
    run(s, "select_target", column="price")
    return s


def test_lightgbm_takes_awkward_column_names():
    s = houses_with_areas()
    r = run(s, "train_models", algorithms=["LightGBM"])["evidence"]["results"][0]
    assert r["model"] == "lgbm" and "cv_mean" in r


def test_catboost_learns_from_a_text_column_the_others_leave_out():
    s = houses_with_areas()
    results = {r["model"]: r for r in run(s, "train_models", algorithms=["xgb", "catboost"])["evidence"]["results"]}
    assert results["catboost"]["cv_mean"] > results["xgb"]["cv_mean"] + 0.2  # location carries most of the price
    exam = run(s, "final_exam", algorithm="catboost")["message"]
    assert "Final exam for CatBoost (catboost)" in exam
    prediction = run(s, "predict", algorithm="catboost", features={"location": "area_3", "sqft": 1200, "Unnamed: 1": 0})
    assert "Prediction:" in prediction["message"]
    assert "location" in [f["feature"] for f in run(s, "explain_model", algorithm="catboost")["evidence"]["chart_data"]]


def test_a_downloaded_catboost_model_says_it_reads_text():
    import pickle
    from app.tools.model import model_bundle
    s = houses_with_areas()
    run(s, "train_models", algorithms=["catboost"])
    bundle = pickle.loads(model_bundle(s, "catboost")[1])
    assert bundle["reads_text"] and "location" in bundle["feature_names"]


# ── Models cells: your own cross-validation and tuning ───────────────────────

@pytest.fixture
def code_on(monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", True)
    monkeypatch.setattr(settings, "CODE_SANDBOX", "local")
    sandbox.reset()
    yield
    sandbox.reset()


TUNED = '''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
cv = KFold(n_splits=4, shuffle=True, random_state=0)
scoring = "neg_mean_absolute_error"
models = {"tuned_rf": GridSearchCV(RandomForestRegressor(random_state=0), {"n_estimators": [20, 40], "max_depth": [3, None]}, cv=3)}
'''


def test_a_models_cell_with_its_own_cv_and_tuning(code_on):
    s = notebook()
    result = run(s, "custom_models", code=TUNED)
    r = result["evidence"]["results"][0]
    assert set(r["best_params"]) == {"n_estimators", "max_depth"} and r["cv_mean"] < 0  # MAE is negative in sklearn
    assert "chose:" in result["message"] and "Not verified" in result["message"] and "KFold" in result["message"]
    assert run(s, "final_exam", algorithm="tuned_rf")["success"]


def test_a_search_with_a_named_scoring_comes_back(code_on):
    """GridSearchCV(scoring="r2") keeps sklearn's r2_score inside it: a scoring function is allowed back."""
    s = notebook()
    code = ("from sklearn.linear_model import Ridge\nfrom sklearn.model_selection import GridSearchCV\n"
            "models = {'ridge_grid': GridSearchCV(Ridge(), {'alpha': [0.1, 10]}, cv=3, scoring='r2')}")
    r = run(s, "custom_models", code=code)["evidence"]["results"][0]
    assert "error" not in r and set(r["best_params"]) == {"alpha"}


def test_shuffled_folds_on_time_data_get_a_warning(code_on):
    s = notebook(time=True)
    code = TUNED.replace('scoring = "neg_mean_absolute_error"\n', "")
    message = run(s, "custom_models", code=code)["message"]
    assert "ordered by time" in message and "learn from the future" in message


def test_time_data_still_trains_after_the_date_column_is_removed():
    """It crashed looking for the date column to order the practice split; the rows are in time order anyway."""
    s = notebook(time=True)
    assert "date" not in s.df.columns
    assert "in time order" in run(s, "train_models", algorithms=["linear", "hgb"])["message"]


# ── Clustering ───────────────────────────────────────────────────────────────

def blobs(n=300):
    rng = np.random.default_rng(1)
    centres = np.array([[0, 0], [6, 6], [0, 8]])
    points = np.vstack([c + rng.normal(0, 0.8, (n // 3, 2)) for c in centres])
    df = pd.DataFrame(points, columns=["x1", "x2"]).sample(frac=1, random_state=0).reset_index(drop=True)
    df["price"] = df.x1 * 10 + df.x2 + rng.normal(0, 1, len(df))
    return df


@pytest.mark.parametrize("method", ["kmeans", "gmm"])
def test_clusters_are_found_and_new_rows_join_the_nearest(method):
    s = DataSession(blobs(), "b.csv")
    run(s, "select_target", column="price")
    message = run(s, "find_clusters", method=method)["message"]
    assert "3 groups" in message and "chosen automatically" in message and "clearly separated" in message
    assert s.recipe[-1].learned["columns"] == ["x1", "x2"]  # never the target
    test = s.prepare(s.test_df)
    assert set(test["cluster"]) == set(s.df["cluster"]) == {0, 1, 2}
    # The same point always lands in the same group
    near = s.prepare(pd.DataFrame({"x1": [6.1, 6.0], "x2": [5.9, 6.2]}))
    assert near["cluster"].nunique() == 1


def test_clustering_refuses_the_target_and_text():
    s = DataSession(blobs().assign(name="x"), "b.csv")
    run(s, "select_target", column="price")
    assert "column you're predicting" in execute_tool("find_clusters", s, {"columns": ["price"]})["message"]
    assert "holds text" in execute_tool("find_clusters", s, {"columns": ["name"]})["message"]


def test_clustering_in_clean_only_mode_and_the_exported_script(tmp_path):
    df = blobs()
    s = DataSession(df, "b.csv")
    run(s, "clean_only")
    run(s, "find_clusters", clusters=3)
    script = pipeline_script(s)
    df.to_csv(tmp_path / "b.csv", index=False)
    (tmp_path / "pipeline.py").write_text(script, encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", "pipeline.py", "b.csv"], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    exported = pd.read_csv(tmp_path / "b_clean.csv")
    assert (exported["cluster"].to_numpy() == s.df["cluster"].to_numpy()).all()  # exactly what the app did


def test_a_row_id_column_gets_a_tip_when_training():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"PassengerId": range(1, 201), "age": rng.integers(1, 80, 200).astype(float),
                       "zip_code": rng.permutation(np.arange(10000, 10200)) * 7})
    df["y"] = df.age * 2 + rng.normal(size=200)
    s = DataSession(df, "p.csv")
    run(s, "select_target", column="y")
    message = run(s, "train_models", algorithms=["linear"])["message"]
    assert "'PassengerId' looks like a row number" in message and "zip_code" not in message


def test_the_server_sends_readable_model_names():
    from fastapi.testclient import TestClient
    from app import app
    with TestClient(app) as client:
        config = client.get("/api/config").json()
    assert config["model_names"]["lr"] == "logistic regression" and "svr" in config["models_for"]["regression"]
    assert "svr" not in config["models_for"]["classification"]
