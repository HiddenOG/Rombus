"""The final exam: one honest score on the locked test rows, compared with a baseline."""

import numpy as np
import pandas as pd
import pytest

from app.session import DataSession
from app.tools import execute_tool


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


@pytest.fixture
def churn():
    rng = np.random.default_rng(0)
    n = 200
    visits = rng.integers(0, 30, n)
    df = pd.DataFrame({
        "visits": visits.astype(float),
        "plan": rng.choice(["basic", "pro"], n),
        "age": np.where(rng.random(n) < 0.1, np.nan, rng.integers(18, 70, n)),
    })
    df["churned"] = np.where(df["visits"] < 10, "yes", "no")  # learnable
    s = DataSession(df, "churn.csv")
    run(s, "select_target", column="churned")
    run(s, "impute_all_missing")
    run(s, "encode_all_categorical", method="onehot")
    return s


def test_training_does_the_practice_split_itself(churn):
    result = run(churn, "train_models", algorithms=["lr"])
    assert "Made a practice split first" in result["message"]


def test_exam_scores_once_and_beats_baseline(churn):
    run(churn, "train_models", algorithms=["dt", "lr"])
    exam = run(churn, "final_exam")["evidence"]["exam"]
    assert exam["test_rows"] == len(churn.test_df)
    assert exam["metric"] == "accuracy"
    assert exam["score"] > exam["baseline"]  # better than always guessing the majority
    assert exam["look"] == 1
    assert sum(map(sum, exam["confusion"])) == exam["test_rows"]
    assert churn.current_stage == "INTERPRET"


def test_test_rows_use_training_sticky_notes(churn):
    # The locked rows were never part of learning: their blanks get the TRAINING mean
    run(churn, "train_models", algorithms=["lr"])
    run(churn, "final_exam")
    train_mean = churn.recipe[0].learned["fill"]["age"]
    prepared = churn.prepare(churn.test_df)
    filled = prepared.loc[churn.test_df["age"].isna(), "age"]
    assert (filled == train_mean).all()


def test_every_extra_look_is_flagged(churn):
    run(churn, "train_models", algorithms=["lr"])
    run(churn, "final_exam")
    second = run(churn, "final_exam")
    assert "look #2" in second["message"]
    assert churn.exam_count == 2


def test_flags_a_model_that_only_guesses_the_common_answer():
    # Pure noise features, 85% "no": the best a model can do is always say "no"
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"noise": rng.random(300), "churned": np.where(rng.random(300) < 0.15, "yes", "no")})
    s = DataSession(df, "noise.csv")
    run(s, "select_target", column="churned")
    run(s, "train_models", algorithms=["lr"])
    result = run(s, "final_exam")
    assert result["evidence"]["exam"]["no_better_than_guessing"]
    assert "no better than just guessing" in result["message"]


def test_prediction_form_fields_come_from_raw_training_rows(churn):
    fields = {f["name"]: f for f in churn.get_summary()["raw_features"]}
    assert set(fields) == {"visits", "plan", "age"}  # original columns, not the one-hot ones; no target
    assert fields["plan"]["kind"] == "category" and set(fields["plan"]["options"]) == {"basic", "pro"}
    assert fields["visits"]["kind"] == "number"
    assert fields["visits"]["typical"] == float(churn.raw_train["visits"].median())

    run(churn, "remove_columns", columns=["age"])
    assert "age" not in {f["name"] for f in churn.get_summary()["raw_features"]}  # removed columns aren't asked for


def test_prediction_includes_confidence(churn):
    run(churn, "train_models", algorithms=["lr", "dt"])
    result = run(churn, "predict", features={"visits": 2, "plan": "basic", "age": 40})
    probs = result["evidence"]["probabilities"]
    assert set(probs) == {"yes", "no"}
    assert abs(sum(probs.values()) - 1) < 1e-6
    assert result["evidence"]["prediction"] == "yes"  # few visits -> churns
    assert "Confidence" in result["message"]


def test_exam_needs_a_trained_model(churn):
    result = execute_tool("final_exam", churn, {})
    assert not result["success"] and "Train at least one model" in result["message"]


def test_regression_exam():
    rng = np.random.default_rng(1)
    x = rng.random(150) * 10
    df = pd.DataFrame({"x": x, "noise": rng.random(150), "price": 3 * x + rng.normal(0, 0.5, 150)})
    s = DataSession(df, "prices.csv")
    run(s, "select_target", column="price")
    run(s, "train_models", algorithms=["ridge"])
    exam = run(s, "final_exam")["evidence"]["exam"]
    assert exam["metric"] == "R²" and exam["score"] > 0.9
    assert exam["baseline"] < 0.1  # guessing the average explains almost nothing
