"""Custom views and custom models — plus the referee cells that stay protected."""

import numpy as np
import pandas as pd
import pytest

from app.action_code import PROTECTED, VIEW_TOOLS, code_info
from app.gemini import build_tool_declarations
from app.session import DataSession
from app.tools import execute_tool


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


@pytest.fixture
def session():
    rng = np.random.default_rng(0)
    n = 160
    visits = rng.integers(0, 30, n).astype(float)
    df = pd.DataFrame({
        "visits": visits,
        "age": rng.integers(18, 70, n).astype(float),
        "plan": rng.choice(["basic", "pro"], n),
        "churned": np.where(visits < 10, "yes", "no"),
    })
    s = DataSession(df, "churn.csv")
    run(s, "select_target", column="churned")
    return s


VIEW_ARGS = {
    "plot_histogram": {"column": "plan"},
    "plot_boxplot": {"column": "age"},
    "plot_scatterplot": {"x": "visits", "y": "age", "hue": "plan"},
}


class TestViews:
    @pytest.mark.parametrize("tool", sorted(VIEW_TOOLS))
    def test_shown_code_runs_as_a_custom_view(self, session, tool):
        info = code_info(session, tool, VIEW_ARGS.get(tool, {}))
        assert info["custom_tool"] == "custom_view"
        result = run(session, "custom_view", code=info["template"])
        assert result.get("evidence"), f"{tool}'s code showed nothing"

    def test_numeric_histogram_code_matches_the_app(self, session):
        info = code_info(session, "plot_histogram", {"column": "age"})
        table = run(session, "custom_view", code=info["template"])["evidence"]["preview"]
        app_bins = run(session, "plot_histogram", column="age")["evidence"]["chart_data"]
        assert [r["rows"] for r in table] == [b["count"] for b in app_bins]

    def test_print_and_charts_are_captured(self, session):
        code = "print('hello')\ntrain['age'].plot.hist()\nresult = 42"
        evidence = run(session, "custom_view", code=code)["evidence"]
        assert "42" in evidence["output"] and "hello" in evidence["output"]
        assert evidence["images"][0].startswith("data:image/png;base64,")

    def test_views_cannot_change_the_training_rows(self, session):
        before = session.df.copy()
        run(session, "custom_view", code="train['age'] = 0\nresult = train.head()")
        pd.testing.assert_frame_equal(session.df, before)


class TestModels:
    def test_custom_models_train_and_take_the_exam(self, session):
        info = code_info(session, "train_models", {"algorithms": ["rf"]})
        assert info["custom_tool"] == "custom_models" and "RandomForestClassifier" in info["template"]
        code = ("from sklearn.ensemble import RandomForestClassifier\n"
                "models = {'deep_forest': RandomForestClassifier(n_estimators=50, max_depth=8, random_state=0)}")
        trained = run(session, "custom_models", code=code)
        assert trained["evidence"]["best_model"] == "deep_forest"
        exam = run(session, "final_exam")["evidence"]["exam"]
        assert exam["model"] == "deep_forest" and exam["score"] > exam["baseline"]

    @pytest.mark.parametrize("code,message", [
        ("x = 1", "must define `models`"),
        ("models = {'bad': 5}", "isn't a model"),
    ])
    def test_bad_models_code(self, session, code, message):
        result = execute_tool("custom_models", session, {"code": code})
        assert not result["success"] and message in result["message"]


class TestReferee:
    @pytest.mark.parametrize("tool", sorted(PROTECTED))
    def test_referee_cells_are_shown_but_not_editable(self, session, tool):
        info = code_info(session, tool, {"column": "churned"} if tool == "select_target" else {})
        assert info["code"] and info["custom_tool"] is None and info["protected"]


def test_ai_is_never_offered_custom_code_tools():
    names = {d["name"] for d in build_tool_declarations()}
    assert not names & {"custom_code", "custom_view", "custom_models"}
