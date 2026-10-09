"""Tests for the recipe engine — written as the examples from the lessons."""

import pandas as pd
import pytest

from app.recipe import Context, Line, apply_to, replay, run_line

CTX = Context(target="label")


def run(df, *lines):
    """Run lines on training rows; return the cleaned rows and the lines (with sticky notes)."""
    lines = [Line(tool, args) for tool, args in lines]
    return replay(df, lines, CTX), lines


class TestStickyNotes:
    def test_median_learned_from_training_rows_only(self):
        # Lesson: training ages 20, 30, 40, blank -> sticky note says 30
        train = pd.DataFrame({"age": [20, 30, 40, None], "label": [1, 0, 1, 0]})
        cleaned, lines = run(train, ("impute_missing", {"column": "age", "strategy": "median"}))
        assert lines[0].learned == {"fill": {"age": 30.0}}
        assert cleaned["age"].tolist() == [20, 30, 40, 30]

        # Test rows 50, 60, blank -> filled with 30 from the sticky note, NOT 55 from the test rows
        test = pd.DataFrame({"age": [50, 60, None], "label": [1, 1, 0]})
        assert apply_to(test, lines, CTX)["age"].tolist() == [50, 60, 30]

    def test_most_common_city(self):
        # Lesson: London, Paris, London, London, blank -> sticky note says London
        train = pd.DataFrame({"city": ["London", "Paris", "London", "London", None], "label": [1, 0, 1, 0, 1]})
        _, lines = run(train, ("impute_missing", {"column": "city", "strategy": "mode"}))
        assert lines[0].learned == {"fill": {"city": "London"}}

        # Even if the test rows are all Paris, a blank test city gets London
        test = pd.DataFrame({"city": ["Paris", "Paris", None], "label": [0, 0, 1]})
        assert apply_to(test, lines, CTX)["city"].tolist() == ["Paris", "Paris", "London"]

    def test_scaling_uses_training_mean_and_spread(self):
        train = pd.DataFrame({"x": [0.0, 10.0], "label": [0, 1]})
        _, lines = run(train, ("scale_all_features", {}))
        assert lines[0].learned == {"scale": {"x": {"mean": 5.0, "std": 5.0}}}
        test = pd.DataFrame({"x": [20.0], "label": [1]})
        assert apply_to(test, lines, CTX)["x"].tolist() == [3.0]  # (20 - 5) / 5

    def test_target_is_never_scaled_or_imputed(self):
        train = pd.DataFrame({"x": [1.0, None, 3.0], "label": [0.0, 1.0, None]})
        cleaned, _ = run(train, ("impute_all_missing", {}), ("scale_all_features", {}))
        assert cleaned["label"].isna().sum() == 1
        assert cleaned["label"].dropna().tolist() == [0.0, 1.0]


class TestNewCustomers:
    def test_the_original_crash_is_fixed(self):
        # The bug from lesson 1: model learned color_blue/green/red, new customer says color: red
        train = pd.DataFrame({
            "size": [1, 2, 3, 4, 5, 6],
            "color": ["red", "blue", "green", "red", "blue", "green"],
            "label": ["yes", "no", "yes", "no", "yes", "no"],
        })
        cleaned, lines = run(train,
                             ("encode_categorical", {"column": "color", "method": "onehot"}),
                             ("scale_all_features", {}))
        new_customer = pd.DataFrame([{"size": 4, "color": "red"}])  # no label: that's what we predict
        prepared = apply_to(new_customer, lines, CTX)

        assert list(prepared.columns) == [c for c in cleaned.columns if c != "label"]
        assert prepared.loc[0, "color_red"] != prepared.loc[0, "color_blue"]

    def test_unseen_category_onehot_is_all_zeros(self):
        train = pd.DataFrame({"city": ["A", "B"], "label": [0, 1]})
        _, lines = run(train, ("encode_categorical", {"column": "city", "method": "onehot"}))
        out = apply_to(pd.DataFrame({"city": ["Z"]}), lines, CTX)
        assert out[["city_A", "city_B"]].values.tolist() == [[0, 0]]

    def test_unseen_category_label_is_minus_one(self):
        train = pd.DataFrame({"city": ["A", "B"], "label": [0, 1]})
        _, lines = run(train, ("encode_categorical", {"column": "city", "method": "label"}))
        assert apply_to(pd.DataFrame({"city": ["B", "Z"]}), lines, CTX)["city"].tolist() == [1, -1]


class TestKinds:
    def test_duplicates_removed_from_training_rows_only(self):
        train = pd.DataFrame({"x": [1, 1, 2], "label": [0, 0, 1]})
        cleaned, lines = run(train, ("remove_duplicates", {}))
        assert len(cleaned) == 2
        # A test set (or a batch of new customers) may legitimately contain identical rows
        assert len(apply_to(pd.DataFrame({"x": [1, 1], "label": [0, 0]}), lines, CTX)) == 2

    def test_fixed_step_needs_no_sticky_note(self):
        train = pd.DataFrame({"customer_id": [1, 2], "x": [3, 4], "label": [0, 1]})
        cleaned, lines = run(train, ("remove_columns", {"columns": ["customer_id"]}))
        assert lines[0].learned == {}
        assert "customer_id" not in cleaned.columns


class TestReplay:
    def test_replay_matches_step_by_step(self):
        train = pd.DataFrame({"age": [20, None, 40], "city": ["a", "b", None], "label": [0, 1, 0]})
        steps = [("impute_all_missing", {}), ("encode_all_categorical", {"method": "onehot"}), ("scale_all_features", {})]

        lines = [Line(t, a) for t, a in steps]
        df = train.copy()
        for line in lines:
            df = run_line(line, df, CTX)

        replayed, _ = run(train, *steps)
        pd.testing.assert_frame_equal(df, replayed)

    def test_replay_never_changes_the_raw_rows(self):
        train = pd.DataFrame({"age": [20, None], "label": [0, 1]})
        run(train, ("impute_missing", {"column": "age"}))
        assert train["age"].isna().sum() == 1


class TestValidation:
    def test_cannot_remove_target(self):
        with pytest.raises(ValueError, match="target column"):
            run(pd.DataFrame({"x": [1], "label": [0]}), ("remove_columns", {"columns": ["label"]}))

    def test_mean_on_text_refused(self):
        with pytest.raises(ValueError, match="non-numeric"):
            run(pd.DataFrame({"c": ["a", None], "label": [0, 1]}), ("impute_missing", {"column": "c", "strategy": "mean"}))

    def test_unknown_column(self):
        with pytest.raises(ValueError, match="not found"):
            run(pd.DataFrame({"x": [1], "label": [0]}), ("impute_missing", {"column": "nope"}))

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            run(pd.DataFrame({"x": [1.0], "label": [0]}), ("impute_missing", {"column": "x", "strategy": "magic"}))
