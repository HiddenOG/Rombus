"""Lesson 1 — how much does data leakage actually matter?

Run from backend/:  python -m experiments.leakage_demo

A. Worst case: selecting features on ALL rows before cross-validation, on pure noise.
B. Our app today: imputing + scaling on ALL rows before the split, on real data.
C. The predict bug: preprocessing applied to the session DataFrame can't be replayed on new input.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)


def part_a():
    print("A. Feature selection on pure noise (100 rows, 5000 random features, random labels)")
    X = rng.normal(size=(100, 5000))
    y = rng.integers(0, 2, size=100)  # labels are coin flips: true accuracy is 50%

    # WRONG: pick the 20 features most correlated with y using ALL rows, then cross-validate
    X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)
    leaky = cross_val_score(LogisticRegression(), X_selected, y, cv=5).mean()

    # RIGHT: selection happens inside each CV fold, seeing only that fold's training rows
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
    honest = cross_val_score(pipe, X, y, cv=5).mean()

    print(f"   leaky accuracy:  {leaky:.1%}   <- looks like a great model")
    print(f"   honest accuracy: {honest:.1%}   <- the truth: it's a coin flip\n")


def part_b():
    print("B. What our app does: impute + scale on ALL rows, then split (breast cancer data, 15% values removed)")
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.mask(rng.random(X.shape) < 0.15)  # knock out 15% of values at random

    leaky_scores, honest_scores = [], []
    for seed in range(50):
        # WRONG (current app): learn mean/std from every row, then split
        X_prep = StandardScaler().fit_transform(SimpleImputer().fit_transform(X))
        Xtr, Xte, ytr, yte = train_test_split(X_prep, y, test_size=0.2, random_state=seed)
        leaky_scores.append(LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte))

        # RIGHT: split raw data first, pipeline learns mean/std from training rows only
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        pipe = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
        honest_scores.append(pipe.fit(Xtr, ytr).score(Xte, yte))

    diff = np.mean(leaky_scores) - np.mean(honest_scores)
    print(f"   leaky accuracy:  {np.mean(leaky_scores):.2%}  (avg of 50 splits)")
    print(f"   honest accuracy: {np.mean(honest_scores):.2%}")
    print(f"   inflation:       {diff:+.2%}\n")


def part_c():
    print("C. The predict bug in our app (fixed by the recipe engine; this used to crash)")
    from app.session import DataSession
    from app.tools import execute_tool

    df = pd.DataFrame({
        "size": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
        "color": ["red", "blue", "green"] * 10,
        "label": ["yes", "no"] * 15,
    })
    s = DataSession(df, "demo.csv")
    for tool, args in [
        ("select_target", {"column": "label"}),
        ("encode_categorical", {"column": "color", "method": "onehot"}),
        ("scale_all_features", {}),
        ("train_test_split", {}),
        ("train_models", {"algorithms": ["lr"]}),
    ]:
        execute_tool(tool, s, args)

    print(f"   model was trained on columns: {list(s.X_train.columns)}")
    raw_row = {"size": 4, "color": "red"}
    result = execute_tool("predict", s, {"features": raw_row})
    print(f"   predict({raw_row}) -> success={result['success']}: {result['message'][:150]}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
