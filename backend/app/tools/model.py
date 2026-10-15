"""Model tools — training, prediction, AutoML, and SHAP explanations."""

import copy
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List, Optional, Tuple

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.base import clone
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from app.tools import register_tool, validate_column, categorical_columns
from app.session import DataSession


# --- Algorithm Registry ---

# XGBoost's settings, written out: its own printout lists every possible setting (mostly None), so the
# Code tab and the editable template show these instead
XGB_SETTINGS = dict(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                    random_state=42, n_jobs=4)
CLASSIFICATION_MODELS = {
    "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "gb": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "lr": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "dt": lambda: DecisionTreeClassifier(random_state=42),
    "knn": lambda: KNeighborsClassifier(),
    "svc": lambda: SVC(kernel='linear', C=1, random_state=42),
    "xgb": lambda: XGBClassifier(**XGB_SETTINGS, eval_metric="logloss"),
}

REGRESSION_MODELS = {
    "linear": lambda: LinearRegression(),
    "rf": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "gb": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "ridge": lambda: Ridge(),
    "lasso": lambda: Lasso(),
    "dt": lambda: DecisionTreeRegressor(random_state=42),
    "knn": lambda: KNeighborsRegressor(),
    "svr": lambda: SVR(kernel='linear', C=1),
    "xgb": lambda: XGBRegressor(**XGB_SETTINGS),
}

# AutoML param grids
AUTOML_CLASSIFICATION_GRIDS = {
    "rf": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    },
    "gb": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2]},
    },
    "lr": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {"C": [0.01, 0.1, 1, 10]},
    },
}

AUTOML_REGRESSION_GRIDS = {
    "rf": {
        "model": RandomForestRegressor(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    },
    "gb": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2]},
    },
    "ridge": {
        "model": Ridge(),
        "params": {"alpha": [0.01, 0.1, 1, 10, 100]},
    },
}


# Long names people (and AI models) naturally write, mapped to the short keys above
# Names of libraries that aren't installed but quietly stand in for a built-in (the message says so).
# Empty on purpose: a silent substitute (xgboost -> gb) confused people and the assistant alike, so an
# unknown library is refused with the list of what's available instead.
STAND_INS: set = set()

ALGORITHM_ALIASES = {
    "random_forest": "rf", "randomforest": "rf",
    "gradient_boosting": "gb", "gradientboosting": "gb", "gbm": "gb",
    "xgboost": "xgb", "xgbregressor": "xgb", "xgbclassifier": "xgb", "extreme_gradient_boosting": "xgb",
    "logistic_regression": "lr", "logisticregression": "lr", "logistic": "lr",
    "decision_tree": "dt", "decisiontree": "dt", "tree": "dt",
    "k_nearest_neighbors": "knn", "nearest_neighbors": "knn", "kneighbors": "knn",
    "svm": "svc", "support_vector_machine": "svc",
    "ridge_regression": "ridge", "lasso_regression": "lasso",
    "linear_regression": "linear", "linearregression": "linear", "ols": "linear", "lr_regression": "linear",
}


def resolve_algorithms(asked: Any, problem_type: Optional[str]) -> Tuple[List[str], List[str]]:
    """What people (and AIs) call models -> registry keys: "Random Forest" -> rf, "xgboost" -> xgb,
    "rf, gb" -> [rf, gb]. Returns (keys, notes about stand-ins). One place, so training and the code
    shown for it always agree."""
    if isinstance(asked, str):
        asked = [a for a in asked.split(",") if a.strip()]
    asked = [str(a).strip() for a in asked]
    keys = [ALGORITHM_ALIASES.get(a.lower().replace(" ", "_").replace("-", "_"), a.lower()) for a in asked]
    if problem_type == "regression":
        # "lr" is logistic regression for yes/no problems; for numbers it can only mean linear regression
        keys = ["linear" if k == "lr" else k for k in keys]
    stand_ins = [f"{a} -> {ALGORITHM_ALIASES[a.lower()]}" for a in asked if a.lower() in STAND_INS]
    return list(dict.fromkeys(keys)), stand_ins

ALGORITHM = {
    "type": "string",
    "description": "Algorithm key of a trained model (see train_models).",
}


# Leftover text columns with more values than this aren't one-hot encoded automatically: a date stored
# as text, an ID or free text would become thousands of columns (and teach the model nothing)
MAX_AUTO_ONEHOT = 100


def ignored_columns(df: pd.DataFrame, target: str) -> List[str]:
    """Columns a model can't use as they are: dates (use Date parts) and text with too many values."""
    from app.expressions import looks_like_dates
    X = df.drop(columns=[target], errors="ignore")
    dates = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    text = categorical_columns(X)
    # Dates stored as text: a column per day means nothing to a model, however few days there are
    text_dates = [c for c in text if len(X) > 1 and looks_like_dates(X[c])]
    wide = [c for c in text if c not in text_dates and X[c].nunique() > MAX_AUTO_ONEHOT]
    return dates + text_dates + wide


def model_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Turn cleaned rows into the all-numeric table a model needs.

    Used at training AND prediction time so both always build features the same way.
    Leftover text columns are one-hot encoded (keeping every category, so a single new row
    still gets its own column) and remaining blanks become 0. Dates and text with too many values
    are left out (see ignored_columns). At prediction time a single row has one value per column,
    so nothing is left out there; columns are lined up with training's afterwards.
    """
    X = df.drop(columns=[target], errors="ignore")
    X = X.drop(columns=ignored_columns(X, target))
    text_cols = categorical_columns(X)
    if text_cols:
        X = pd.get_dummies(X, columns=text_cols, dtype=int)
    return X.fillna(0)


def target_classes(session: DataSession) -> Optional[List[str]]:
    """If the target was label-encoded in the recipe, the classes needed to decode predictions."""
    for line in reversed(session.recipe):
        if line.tool in ("encode_categorical", "encode_all_categorical") and line.args.get("method", "label") == "label":
            classes = line.learned.get("classes", {}).get(session.target_column)
            if classes:
                return classes
    return None


def _xgb_label_problem(session: DataSession) -> Optional[str]:
    """XGBoost's classifier needs the answers numbered 0, 1, 2...; why it can't run here, or None."""
    labels = pd.Series(session.original_df[session.target_column].dropna().unique())
    numbered = pd.api.types.is_numeric_dtype(labels) and sorted(labels.tolist()) == list(range(len(labels)))
    if numbered or target_classes(session):
        return None
    return (f"XGBoost needs '{session.target_column}' as numbers 0, 1, 2... (it has {', '.join(map(str, labels[:4]))}). "
            f"Label-encode it first (Encode categorical: {session.target_column}, label): predictions still come "
            "back as the original labels. The other models work as it is.")


def fit_and_score(session: DataSession, name: str, model: Any, cv_folds: int) -> Dict[str, Any]:
    """Train one model on the practice split, score it, and remember it. Shared by built-in and custom models."""
    model.fit(session.X_train, session.y_train)
    y_pred = model.predict(session.X_test)

    if session.problem_type == "regression":
        result = {
            "model": name,
            "r2_score": round(float(r2_score(session.y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(session.y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(session.y_test, y_pred))), 4),
        }
    else:
        result = {
            "model": name,
            "accuracy": round(float(accuracy_score(session.y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(session.y_test, y_pred, average="weighted", zero_division=0)), 4),
            "report": classification_report(session.y_test, y_pred, zero_division=0),
        }

    # Cross-validation: several mini-exams within the training part
    cv_scores = cross_val_score(model, session.X_train, session.y_train, cv=cv_splitter(session, cv_folds))
    result["cv_mean"] = round(float(cv_scores.mean()), 4)
    result["cv_std"] = round(float(cv_scores.std()), 4)

    session.trained_models[name] = model
    session.model_results[name] = result
    return result


def finish_training(session: DataSession, results: List[Dict[str, Any]]) -> Optional[str]:
    """Pick the best result by practice score and move the stage on."""
    scored = [r for r in results if "cv_mean" in r]
    if session.current_stage in ("ACQUIRE", "SCRUB", "EXPLORE"):
        session.current_stage = "MODEL"
    return max(scored, key=lambda r: r["cv_mean"])["model"] if scored else None


def best_model_name(session: DataSession) -> str:
    """The trained model with the best practice (cross-validation) score."""
    scored = {a: r.get("cv_mean", float("-inf")) for a, r in session.model_results.items() if a in session.trained_models}
    return max(scored, key=scored.get) if scored else next(iter(session.trained_models))


def ensure_practice_split(session: DataSession) -> str:
    """Do the default practice split if there isn't one yet; returns a note when it did."""
    if session.X_train is not None:
        return ""
    if not session.target_column:
        raise ValueError("Choose the target column first.")
    result = do_train_test_split(session, {"stratify": session.problem_type == "classification"})
    if not result["success"]:
        raise ValueError(result["message"])
    notes = []
    if session.time_column:
        notes.append("in time order: the check rows are the latest ones")
    skipped = ignored_columns(session.df, session.target_column)
    if skipped:
        notes.append(f"Not used as model inputs: {skipped} (dates, or text with over {MAX_AUTO_ONEHOT} values)")
    return (f"(Made a practice split first: {len(session.X_train)} rows to learn from, {len(session.X_test)} to "
            f"check against{'; ' + '. '.join(notes) if notes else ''}.)")


def safe_cv_folds(session: DataSession, requested: int) -> int:
    """Cross-validation folds, capped so every fold can contain every class."""
    folds = min(int(requested), len(session.X_train))
    if session.problem_type == "classification":
        counts = session.y_train.value_counts()
        if counts.min() < 2:
            rare = ", ".join(f"'{c}'" for c in counts[counts < 2].index[:5])
            raise ValueError(
                f"Some answers appear only once in the rows the models learn from ({rare}), and checking a model "
                f"needs at least 2 of each. Get more rows, or if '{session.target_column}' is really a number, "
                "predict it as one: change its type to number before choosing it as the target.")
        folds = min(folds, int(counts.min()))
    return max(folds, 2)


@register_tool(
    "train_test_split",
    "Practice split: divide the training rows into a part to learn from and a part to check against. "
    "The locked test rows are not used.",
    {
        "test_size": {"type": "number", "description": "Fraction for the test set (default 0.2)."},
        "stratify": {"type": "boolean", "description": "Stratify by target (classification only)."},
    },
    kind="action",
)
def do_train_test_split(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Split data into training and test sets."""
    if not session.target_column:
        return {"success": False, "message": "Select a target column first (use 'select_target' tool)."}

    validate_column(session, session.target_column)

    test_size = args.get("test_size", 0.2)
    stratify = args.get("stratify", False)

    X = model_features(session.df, session.target_column)
    y = session.df[session.target_column]
    skipped = ignored_columns(session.df, session.target_column)
    object_cols = [c for c in categorical_columns(session.df.drop(columns=[session.target_column])) if c not in skipped]

    if session.time_column:
        # Time order: learn from the earlier training rows, check against the later ones (never the reverse)
        order = time_order(session.df, session.time_column)
        X, y = X.loc[order], y.loc[order]
        session.X_train, session.X_test, session.y_train, session.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False)
    else:
        # Stratifying needs at least 2 rows of every class; with a rarer class, split without it
        can_stratify = session.problem_type == "classification" and y.value_counts().min() >= 2
        stratify_param = y if stratify and can_stratify else None
        session.X_train, session.X_test, session.y_train, session.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_param
        )

    # TODO(piece 3): evaluate on the locked test rows instead of this practice split
    msg = (
        f"Practice split of the training rows: {len(session.X_train)} to learn from, "
        f"{len(session.X_test)} to check against (test_size={test_size}). "
        f"The {len(session.test_df) if session.is_split else 0} locked test rows stay untouched."
    )
    if session.time_column:
        msg += f" In time order of '{session.time_column}': the check rows are the latest ones."
    if object_cols:
        msg += f" Auto-encoded {len(object_cols)} categorical columns: {object_cols}."
    if skipped:
        msg += (f" Not used as model inputs: {skipped} (dates, or text with over {MAX_AUTO_ONEHOT} values; "
                "use Date parts, Group rare categories, or remove them).")

    return {"success": True, "message": msg}


def time_order(df: pd.DataFrame, time_column: str) -> pd.Index:
    """Row labels sorted by time (stable, so rows with the same time keep their order)."""
    from app.expressions import as_dates
    return as_dates(df[time_column]).sort_values(kind="stable").index


def cv_splitter(session: DataSession, folds: int):
    """Cross-validation folds: shuffled for ordinary data; for time data, always train on the past
    and test on the following stretch (TimeSeriesSplit), never the other way round."""
    if session.time_column:
        from sklearn.model_selection import TimeSeriesSplit
        return TimeSeriesSplit(n_splits=folds)
    return folds


@register_tool(
    "train_models",
    "Train and cross-validate models. Classification algorithms: "
    + ", ".join(CLASSIFICATION_MODELS) + ". Regression algorithms: " + ", ".join(REGRESSION_MODELS) + ".",
    {
        "algorithms": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(set(CLASSIFICATION_MODELS) | set(REGRESSION_MODELS))},
        },
        "cv_folds": {"type": "integer", "description": "Cross-validation folds (default 5)."},
    },
    kind="action",
)
def train_models(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Train one or more ML models (doing the practice split first if it hasn't happened)."""
    model_registry = CLASSIFICATION_MODELS if session.problem_type == "classification" else REGRESSION_MODELS
    alg_names, stand_ins = resolve_algorithms(args.get("algorithms", ["rf", "gb"]), session.problem_type)
    unknown = [a for a in alg_names if a not in model_registry]
    if unknown:
        return {"success": False,
                "message": f"Unknown algorithm{'s' if len(unknown) > 1 else ''} for {session.problem_type}: "
                           f"{', '.join(unknown)}. Available: {', '.join(model_registry)}."}
    if "xgb" in alg_names and session.problem_type == "classification":
        xgb_problem = _xgb_label_problem(session)
        if xgb_problem:
            return {"success": False, "message": xgb_problem}

    auto_split = ensure_practice_split(session)
    cv_folds = safe_cv_folds(session, args.get("cv_folds", 5))
    results = [fit_and_score(session, alg, model_registry[alg](), cv_folds) for alg in alg_names]

    best = finish_training(session, results)
    msg = f"Trained {len(results)} model{'s' if len(results) != 1 else ''}. Best on the practice split: {best}."
    if auto_split:
        msg = f"{auto_split} {msg}"
    if stand_ins:
        msg += f" Not installed here, so the closest built-in was used: {', '.join(stand_ins)}."
    if best:
        msg += " When you're happy with the cleaning and the model, run the final exam on the locked test rows."
    return {
        "success": True,
        "message": msg,
        "evidence": {"results": results, "best_model": best},
    }


@register_tool("auto_ml", "Grid-search several algorithms with hyperparameter tuning and report the best.", kind="action")
def auto_ml(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Automatically find the best model with hyperparameter tuning (AutoML)."""
    ensure_practice_split(session)

    grids = AUTOML_CLASSIFICATION_GRIDS if session.problem_type == "classification" else AUTOML_REGRESSION_GRIDS
    scoring = "accuracy" if session.problem_type == "classification" else "r2"
    folds = safe_cv_folds(session, 3)

    results = []
    for name, config in grids.items():
        try:
            # n_jobs=1: on a shared server, one person's search mustn't take every CPU core
            grid_search = GridSearchCV(clone(config["model"]), config["params"],
                                       cv=cv_splitter(session, folds), scoring=scoring, n_jobs=1, error_score="raise")
            grid_search.fit(session.X_train, session.y_train)
        except Exception as e:
            results.append({"model": name, "error": str(e).splitlines()[0][:200]})
            continue

        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(session.X_test)
        cv = round(float(grid_search.best_score_), 4)
        result = {"model": name, "best_params": grid_search.best_params_, "cv_best_score": cv,
                  # cv_mean: the score every later step (explain, predict, final exam) ranks models by
                  "cv_mean": cv, "cv_std": round(float(grid_search.cv_results_["std_test_score"][grid_search.best_index_]), 4)}
        if session.problem_type == "regression":
            result["r2_score"] = round(float(r2_score(session.y_test, y_pred)), 4)
        else:
            result["accuracy"] = round(float(accuracy_score(session.y_test, y_pred)), 4)

        session.trained_models[name] = best_estimator
        session.model_results[name] = result
        results.append(result)

    best = finish_training(session, results)
    if best is None:
        return {"success": False,
                "message": "AutoML couldn't fit any model: " + "; ".join(f"{r['model']}: {r['error']}" for r in results)}
    overall = best_model_name(session)
    msg = f"AutoML complete. Best tuned model: {best} (CV score: {session.model_results[best]['cv_mean']:.4f})."
    if overall == best:
        msg += " It's now the default for explain, predict and the final exam."
    else:
        msg += (f" Your earlier '{overall}' still has the best practice score "
                f"({session.model_results[overall]['cv_mean']:.4f}), so it stays the default.")
    return {"success": True, "message": msg, "evidence": {"results": results, "best_model": best}}


@register_tool(
    "final_exam",
    "Score a trained model ONCE on the locked test rows (the honest score). Retrains it on all training "
    "rows first. Only suggest this when the user is done cleaning and choosing a model; every extra look "
    "at the test rows makes the score less trustworthy.",
    {"algorithm": {"type": "string", "description": "Trained model to examine (default: best practice score)."}},
    kind="action",
)
def final_exam(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Retrain on all training rows, then score once on the locked test rows prepared by the recipe."""
    if not session.is_split:
        return {"success": False, "message": "Choose the target first: there are no locked test rows yet."}
    if not session.trained_models:
        return {"success": False, "message": "Train at least one model first."}

    name = args.get("algorithm") or best_model_name(session)
    if name not in session.trained_models:
        return {"success": False, "message": f"Model '{name}' isn't trained. Trained: {list(session.trained_models)}."}

    target = session.target_column
    # 1. The final model learns from ALL training rows (the practice split was only for choosing)
    X_all = model_features(session.df, target)
    y_all = session.df[target]
    try:
        fresh = clone(session.trained_models[name])
    except TypeError:  # custom models that don't follow scikit-learn's conventions
        fresh = copy.deepcopy(session.trained_models[name])
    model = fresh.fit(X_all, y_all)

    # 2. The locked rows go through the recipe with the TRAINING sticky notes: no learning from them
    prepared = session.prepare(session.test_df)
    X_test = model_features(prepared, target).reindex(columns=X_all.columns, fill_value=0)
    y_test = prepared[target]
    predicted = model.predict(X_test)

    session.exam_count += 1
    session.trained_models[name] = model  # predictions now use the model trained on all rows
    if session.current_stage in ("ACQUIRE", "SCRUB", "EXPLORE", "MODEL"):
        session.current_stage = "INTERPRET"

    exam: Dict[str, Any] = {
        "model": name,
        "test_rows": int(len(y_test)),
        "look": session.exam_count,
        "practice": session.model_results.get(name, {}).get("cv_mean"),
        "problem": session.problem_type,
    }
    if session.problem_type == "classification":
        majority = y_all.mode().iloc[0]
        labels = sorted(pd.unique(pd.concat([y_test, pd.Series(predicted)]).astype(str)))
        matrix = confusion_matrix(y_test.astype(str), pd.Series(predicted).astype(str), labels=labels)
        exam |= {
            "metric": "accuracy",
            "score": round(float(accuracy_score(y_test, predicted)), 4),
            "baseline": round(float((y_test == majority).mean()), 4),
            "baseline_label": f"always guessing '{majority}'",
            "f1": round(float(f1_score(y_test, predicted, average="weighted", zero_division=0)), 4),
            "labels": labels,
            "confusion": matrix.tolist(),
        }
    else:
        mean = float(y_all.mean())
        exam |= {
            "metric": "R²",
            "score": round(float(r2_score(y_test, predicted)), 4),
            "baseline": round(float(r2_score(y_test, np.full(len(y_test), mean))), 4),
            "baseline_label": f"always guessing the training average ({mean:.3g})",
            "mae": round(float(mean_absolute_error(y_test, predicted)), 4),
            "baseline_mae": round(float(mean_absolute_error(y_test, np.full(len(y_test), mean))), 4),
        }

    exam["no_better_than_guessing"] = exam["score"] <= exam["baseline"] + 0.01
    msg = (
        f"Final exam for {name}: {exam['metric']} {exam['score']} on {exam['test_rows']} locked rows it never saw "
        f"(baseline, {exam['baseline_label']}: {exam['baseline']})."
    )
    if exam["no_better_than_guessing"]:
        msg += (f" Warning: the model did no better than the baseline ({exam['baseline_label']}), so it hasn't "
                "learned anything useful yet. Try other models, more informative columns, or more data"
                + (", or balancing the classes." if session.problem_type == "classification" else "."))
    if exam["practice"] is not None and exam["practice"] - exam["score"] > 0.05:
        msg += f" That's well below the practice score ({exam['practice']}): the model may be overfitting."
    if session.exam_count > 1:
        msg += (f" Note: this is look #{session.exam_count} at the test rows. Every extra look makes this score less "
                f"trustworthy, because choices start being tuned to the exam.")
    return {"success": True, "message": msg, "evidence": {"exam": exam}}


@register_tool(
    "predict",
    "Predict the target for one row of feature values using a trained model.",
    {
        "features": {"type": "object", "description": "Mapping of feature column name to value."},
        "algorithm": ALGORITHM,
    },
    ["features"],
)
def predict(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Make a prediction using a trained model."""
    if not session.trained_models:
        return {"success": False, "message": "Train models first."}

    model_name = args.get("algorithm") or best_model_name(session)
    if model_name not in session.trained_models:
        return {"success": False, "message": f"Model '{model_name}' not found. Available: {list(session.trained_models.keys())}"}

    model = session.trained_models[model_name]
    features = args.get("features", {})

    if not features:
        return {"success": False, "message": "Provide 'features' dict with column values."}

    # The new row goes through the same recipe (with the training sticky notes) as the training rows,
    # then gets the exact columns the model was trained on. It's given in the ORIGINAL columns; a column
    # renamed by the recipe may be given by its new name ('bhk' for 'size').
    raw_columns = [c for c in (session.raw_train if session.raw_train is not None else session.df).columns
                   if c != session.target_column]
    original_name = {}
    for line in reversed(session.recipe):
        if line.tool == "rename_column":
            new, old = str(line.args.get("new_name", "")).strip(), line.args.get("column")
            original_name[new] = original_name.get(old, old)
    features = {original_name.get(k, k) if k not in raw_columns else k: v for k, v in features.items()}
    ignored = [k for k in features if k not in raw_columns]
    removed = {c for line in session.recipe if line.tool == "remove_columns" for c in line.args.get("columns", [])}
    missing = [c for c in raw_columns if c not in features and c not in removed]
    prepared = session.prepare(pd.DataFrame([{k: v for k, v in features.items() if k in raw_columns}]))
    X = model_features(prepared, session.target_column).reindex(columns=session.X_train.columns, fill_value=0)

    prediction = model.predict(X)[0]
    classes = target_classes(session)
    if classes is not None and 0 <= int(prediction) < len(classes):
        prediction = classes[int(prediction)]
    prediction = prediction.item() if hasattr(prediction, "item") else prediction

    # How sure is the model? (classification models that can say)
    probabilities = None
    if session.problem_type == "classification" and hasattr(model, "predict_proba"):
        labels = [classes[int(c)] if classes is not None and 0 <= int(c) < len(classes) else c for c in model.classes_]
        probabilities = {str(label): round(float(p), 4) for label, p in zip(labels, model.predict_proba(X)[0])}

    msg = f"Prediction: {prediction} (using {model_name})."
    if probabilities:
        msg += f" Confidence: {probabilities.get(str(prediction), 0):.0%}."
    if missing:
        msg += f" Not provided, so treated as blank: {', '.join(missing)}."
    if ignored:
        msg += (f" Ignored (not columns of the original data): {', '.join(map(str, ignored))}. Give values for "
                f"the original columns: the recipe makes the rest.")
    return {
        "success": True,
        "message": msg,
        "evidence": {
            "prediction": str(prediction),
            "model_used": model_name,
            "input_features": features,
            "probabilities": probabilities,
        },
    }


@register_tool("explain_model", "Show which features drive a trained model's predictions.", {"algorithm": ALGORITHM})
def explain_model(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate SHAP feature importance explanations."""
    if not session.trained_models:
        return {"success": False, "message": "Train models first."}

    model_name = args.get("algorithm") or best_model_name(session)
    if model_name not in session.trained_models:
        return {"success": False, "message": f"Model '{model_name}' not found."}

    model = session.trained_models[model_name]

    try:
        import shap

        # Use appropriate explainer
        if hasattr(model, 'feature_importances_'):
            # Tree-based: use built-in feature importance + SHAP
            feature_names = session.X_train.columns.tolist()
            importances = model.feature_importances_
            importance_data = sorted(
                [{"feature": f, "importance": round(float(imp), 4)}
                 for f, imp in zip(feature_names, importances)],
                key=lambda x: abs(x["importance"]),
                reverse=True,
            )

            return {
                "success": True,
                "message": f"Feature importance for {model_name}.",
                "evidence": {
                    "chart_type": "bar",
                    "chart_data": importance_data[:15],
                    "title": f"Feature Importance ({model_name})",
                    "x_key": "feature",
                    "y_key": "importance",
                },
            }
        else:
            # For non-tree models, compute permutation importance
            from sklearn.inspection import permutation_importance
            perm_result = permutation_importance(model, session.X_test, session.y_test, n_repeats=10, random_state=42)
            feature_names = session.X_train.columns.tolist()
            importance_data = sorted(
                [{"feature": f, "importance": round(float(imp), 4)}
                 for f, imp in zip(feature_names, perm_result.importances_mean)],
                key=lambda x: abs(x["importance"]),
                reverse=True,
            )

            return {
                "success": True,
                "message": f"Permutation importance for {model_name}.",
                "evidence": {
                    "chart_type": "bar",
                    "chart_data": importance_data[:15],
                    "title": f"Permutation Importance ({model_name})",
                    "x_key": "feature",
                    "y_key": "importance",
                },
            }

    except ImportError:
        return {"success": False, "message": "SHAP library not installed. Run: pip install shap"}
    except Exception as e:
        return {"success": False, "message": f"Could not explain model: {str(e)}"}


def model_bundle(session: DataSession, model_name: Optional[str] = None) -> Tuple[str, bytes]:
    """A trained model with everything needed to use it, as a pickle file to download:
    the model, the recipe with its sticky notes (to prepare new rows), and the feature names."""
    if not session.trained_models:
        raise ValueError("No trained models yet. Train a model first.")
    model_name = model_name or best_model_name(session)
    if model_name not in session.trained_models:
        raise ValueError(f"Model '{model_name}' not found. Trained: {', '.join(session.trained_models)}.")
    return model_name, pickle.dumps({
        "model": session.trained_models[model_name],
        # The recipe with its sticky notes: replay with app.recipe.apply_to before predicting
        "recipe": [{"tool": l.tool, "args": l.args, "learned": l.learned} for l in session.recipe],
        "target": session.target_column,
        "problem_type": session.problem_type,
        "feature_names": session.X_train.columns.tolist() if session.X_train is not None else [],
    })


@register_tool("save_model", "Prepare a trained model (with its preprocessing) for download.", {"algorithm": ALGORITHM}, kind="action")
def save_model(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Check the model can be packaged; the file itself is downloaded from the Export menu
    (it goes to the person's computer, never onto the server's disk)."""
    model_name, _ = model_bundle(session, args.get("algorithm"))
    return {
        "success": True,
        "message": f"Model '{model_name}' is ready: download it from Export → Trained model (.pkl). "
                   "It includes the recipe, so new rows can be prepared exactly like the training rows.",
    }
