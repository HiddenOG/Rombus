"""Code views for actions and views: every tool has one, it's valid Python, and the split code is exact."""

import ast

import numpy as np
import pandas as pd
import pytest

from app.action_code import BUILDERS, action_code
from app.recipe import STEP_TYPES
from app.session import DataSession
from app.tools import TOOL_REGISTRY, execute_tool


@pytest.fixture
def session():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, 120).astype(float),
        "city": rng.choice(["a", "b", "c"], 120),
        "label": rng.choice(["yes", "no"], 120),
    })
    s = DataSession(df, "people.csv")
    assert execute_tool("select_target", s, {"column": "label", "test_size": 0.3})["success"]
    return s


def test_every_non_recipe_tool_has_a_code_view():
    assert set(TOOL_REGISTRY) - set(STEP_TYPES) <= set(BUILDERS)


ARGS = {
    "select_target": {"column": "label", "test_size": 0.3},
    "plot_histogram": {"column": "age"},
    "plot_boxplot": {"column": "age"},
    "plot_scatterplot": {"x": "age", "y": "age", "hue": "city"},
    "train_models": {"algorithms": ["rf", "lr"]},
    "predict": {"features": {"age": 30, "city": "a"}},
}


@pytest.mark.parametrize("tool", sorted(BUILDERS))
def test_code_is_valid_python(session, tool):
    ast.parse(action_code(session, tool, ARGS.get(tool, {})))


def test_split_code_locks_exactly_the_same_rows(session, tmp_path, monkeypatch):
    # Run the shown code on the original file: it must lock away the very same rows the app did
    monkeypatch.chdir(tmp_path)
    session.original_df.to_csv(session.filename, index=False)
    scope = {}
    exec(action_code(session, "select_target", {"column": "label", "test_size": 0.3}), scope)
    shown_test = scope["test"].reset_index(drop=True)
    pd.testing.assert_frame_equal(shown_test, session.test_df, check_dtype=False)


def test_model_code_names_the_real_models(session):
    code = action_code(session, "train_models", {"algorithms": ["rf", "lr"]})
    assert "RandomForestClassifier(random_state=42)" in code
    assert "LogisticRegression(" in code
    assert "from sklearn.ensemble import RandomForestClassifier" in code
