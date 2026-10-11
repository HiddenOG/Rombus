"""Tests for tool functions."""

import pytest
import pandas as pd
import numpy as np
from app.session import DataSession


@pytest.fixture
def sample_session():
    """Create a sample session with the Iris-like dataset."""
    df = pd.DataFrame({
        'Sepal_Length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
                        7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2],
        'Sepal_Width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
                       3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7],
        'Petal_Length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
                       4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9],
        'Species': ['setosa'] * 10 + ['versicolor'] * 10,
    })
    return DataSession(df, "test_iris.csv")


@pytest.fixture
def session_with_missing():
    """Session with missing values."""
    df = pd.DataFrame({
        'A': [1.0, 2.0, None, 4.0, 5.0],
        'B': ['x', None, 'y', 'x', 'z'],
        'C': [10, 20, 30, 40, 50],
    })
    return DataSession(df, "test_missing.csv")


class TestAcquireTools:
    def test_select_target(self, sample_session):
        from app.tools.acquire import select_target
        result = select_target(sample_session, {"column": "Species"})
        assert result["success"]
        assert sample_session.target_column == "Species"
        assert sample_session.problem_type == "classification"

    def test_select_target_missing_column(self, sample_session):
        from app.tools.acquire import select_target
        with pytest.raises(ValueError, match="not found"):
            select_target(sample_session, {"column": "NonExistent"})

    def test_preview_data(self, sample_session):
        from app.tools.acquire import preview_data
        result = preview_data(sample_session, {"rows": 3})
        assert result["success"]
        assert len(result["evidence"]["preview"]) == 3

    def test_get_data_info(self, sample_session):
        from app.tools.acquire import get_data_info
        result = get_data_info(sample_session, {})
        assert result["success"]
        assert "info" in result["evidence"]
        assert "describe" in result["evidence"]


def call(tool, session, args):
    """Call a tool directly (bypassing the lock gate, which test_split.py covers)."""
    from app.tools import TOOL_REGISTRY
    return TOOL_REGISTRY[tool](session, args)


class TestScrubTools:
    def test_remove_columns(self, sample_session):
        result = call("remove_columns", sample_session, {"columns": ["Sepal_Width"]})
        assert result["success"]
        assert "Sepal_Width" not in sample_session.df.columns

    def test_check_duplicates(self, sample_session):
        result = call("check_duplicates", sample_session, {})
        assert result["success"]
        assert "duplicate_count" in result["evidence"]

    def test_check_duplicates_is_not_a_recipe_step(self, sample_session):
        call("check_duplicates", sample_session, {})
        assert sample_session.recipe == []

    def test_impute_missing_mean(self, session_with_missing):
        result = call("impute_missing", session_with_missing, {"column": "A", "strategy": "mean"})
        assert result["success"]
        assert session_with_missing.df["A"].isnull().sum() == 0

    def test_impute_missing_mode(self, session_with_missing):
        result = call("impute_missing", session_with_missing, {"column": "B", "strategy": "mode"})
        assert result["success"]
        assert session_with_missing.df["B"].isnull().sum() == 0

    def test_impute_missing_invalid_strategy(self, session_with_missing):
        with pytest.raises(ValueError, match="Unknown strategy"):
            call("impute_missing", session_with_missing, {"column": "A", "strategy": "mean_on_text"})
        assert session_with_missing.recipe == []  # a failed step leaves no trace

    def test_encode_categorical_label(self, sample_session):
        result = call("encode_categorical", sample_session, {"column": "Species", "method": "label"})
        assert result["success"]
        assert pd.api.types.is_integer_dtype(sample_session.df["Species"])

    def test_encode_categorical_onehot(self, sample_session):
        result = call("encode_categorical", sample_session, {"column": "Species", "method": "onehot"})
        assert result["success"]
        assert "Species" not in sample_session.df.columns

    def test_scale_features(self, sample_session):
        sample_session.target_column = "Species"
        result = call("scale_features", sample_session, {})
        assert result["success"]
        assert abs(sample_session.df["Sepal_Length"].mean()) < 0.1

    def test_undo_after_remove(self, sample_session):
        from app.tools.acquire import select_target
        select_target(sample_session, {"column": "Species"})
        original_cols = list(sample_session.df.columns)
        call("remove_columns", sample_session, {"columns": ["Sepal_Width"]})
        assert "Sepal_Width" not in sample_session.df.columns

        assert sample_session.undo().tool == "remove_columns"
        assert list(sample_session.df.columns) == original_cols

    def test_undo_removes_the_right_step_after_a_chart(self, sample_session):
        # The old bug: charts were logged as steps, so undo named the wrong one
        from app.tools.acquire import select_target
        select_target(sample_session, {"column": "Species"})
        call("impute_all_missing", sample_session, {})
        call("plot_histogram", sample_session, {"column": "Sepal_Length"})
        result = call("undo", sample_session, {})
        assert "impute all missing" in result["message"]
        assert sample_session.recipe == []

    def test_column_validation(self, sample_session):
        with pytest.raises(ValueError, match="not found"):
            call("impute_missing", sample_session, {"column": "FakeColumn", "strategy": "mean"})


class TestExploreTools:
    def test_plot_histogram_numeric(self, sample_session):
        from app.tools.explore import plot_histogram
        result = plot_histogram(sample_session, {"column": "Sepal_Length"})
        assert result["success"]
        assert result["evidence"]["chart_type"] == "histogram"
        assert len(result["evidence"]["chart_data"]) > 0

    def test_plot_histogram_categorical(self, sample_session):
        from app.tools.explore import plot_histogram
        result = plot_histogram(sample_session, {"column": "Species"})
        assert result["success"]
        assert result["evidence"]["chart_type"] == "bar"

    def test_plot_boxplot(self, sample_session):
        from app.tools.explore import plot_boxplot
        result = plot_boxplot(sample_session, {"column": "Sepal_Length"})
        assert result["success"]
        assert "q1" in result["evidence"]["chart_data"]
        assert "median" in result["evidence"]["chart_data"]

    def test_plot_correlation(self, sample_session):
        from app.tools.explore import plot_correlation
        result = plot_correlation(sample_session, {})
        assert result["success"]
        assert result["evidence"]["chart_type"] == "heatmap"

    def test_plot_scatterplot(self, sample_session):
        from app.tools.explore import plot_scatterplot
        result = plot_scatterplot(sample_session, {"x": "Sepal_Length", "y": "Sepal_Width"})
        assert result["success"]
        assert result["evidence"]["chart_type"] == "scatter"


class TestModelTools:
    def test_train_test_split(self, sample_session):
        from app.tools.acquire import select_target
        from app.tools.model import do_train_test_split

        select_target(sample_session, {"column": "Species"})
        result = do_train_test_split(sample_session, {"test_size": 0.3})
        assert result["success"]
        assert sample_session.X_train is not None
        assert sample_session.X_test is not None

    def test_train_models(self, sample_session):
        from app.tools.acquire import select_target
        from app.tools.model import do_train_test_split, train_models

        select_target(sample_session, {"column": "Species"})
        do_train_test_split(sample_session, {"test_size": 0.3})
        result = train_models(sample_session, {"algorithms": ["rf", "dt"]})
        assert result["success"]
        assert len(result["evidence"]["results"]) == 2
        assert "rf" in sample_session.trained_models

    def test_train_unknown_algorithm(self, sample_session):
        from app.tools.acquire import select_target
        from app.tools.model import do_train_test_split, train_models

        select_target(sample_session, {"column": "Species"})
        do_train_test_split(sample_session, {"test_size": 0.3})
        result = train_models(sample_session, {"algorithms": ["unknown_model", "rf"]})
        # Refused as a whole, naming the bad one and what's available (not a "success" that trained nothing)
        assert not result["success"]
        assert "unknown_model" in result["message"] and "rf" in result["message"]
        assert not sample_session.trained_models

    def test_train_accepts_long_algorithm_names(self, sample_session):
        from app.tools.acquire import select_target
        from app.tools.model import train_models

        select_target(sample_session, {"column": "Species"})
        result = train_models(sample_session, {"algorithms": ["Random Forest", "logistic_regression"]})
        assert result["success"], result["message"]
        assert set(sample_session.trained_models) == {"rf", "lr"}
        assert "Trained 2 models" in result["message"]


class TestWorkflowTools:
    def test_transition_stage(self, sample_session):
        from app.tools.workflow import transition_stage
        assert sample_session.current_stage == "ACQUIRE"
        result = transition_stage(sample_session, {})
        assert result["success"]
        assert sample_session.current_stage == "SCRUB"

    def test_transition_stage_final(self, sample_session):
        from app.tools.workflow import transition_stage
        sample_session.current_stage = "COMMUNICATE"
        result = transition_stage(sample_session, {})
        assert result["success"]
        assert "final stage" in result["message"].lower() or "complete" in result["message"].lower()

    def test_undo_no_snapshots(self, sample_session):
        from app.tools.workflow import undo
        result = undo(sample_session, {})
        assert not result["success"]


class TestSessionManagement:
    def test_session_creation(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        session = DataSession(df, "test.csv")
        assert session.filename == "test.csv"
        assert session.current_stage == "ACQUIRE"

    def test_recipe_and_undo_replay(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0, 5.0], "y": [0, 1, 0, 1, 0]})
        session = DataSession(df, "test.csv")
        session.target_column, session.problem_type = "y", "regression"
        session.lock_test_rows(0.2)
        before = session.df.copy()

        session.add_step("impute_missing", {"column": "a"})
        session.add_step("scale_all_features", {})
        assert session.undo_count == 2

        session.undo()
        session.undo()
        pd.testing.assert_frame_equal(session.df, before)
        assert session.undo() is None

    def test_summary_includes_all_columns(self):
        df = pd.DataFrame({f"col_{i}": range(10) for i in range(15)})
        session = DataSession(df, "wide.csv")
        summary = session.get_summary()
        assert len(summary["columns"]) == 15  # All columns, not just 10!

    def test_compact_column_summary(self):
        df = pd.DataFrame({"name": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
        session = DataSession(df, "test.csv")
        summary = session.get_compact_column_summary()
        assert "name" in summary
        assert "value" in summary
