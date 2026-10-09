"""Model tools — training, prediction, AutoML, and SHAP explanations."""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, classification_report,
)

from app.tools import register_tool, validate_column, categorical_columns
from app.session import DataSession
from app.config import settings


# --- Algorithm Registry ---
CLASSIFICATION_MODELS = {
    "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "gb": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "lr": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "dt": lambda: DecisionTreeClassifier(random_state=42),
    "knn": lambda: KNeighborsClassifier(),
    "svc": lambda: SVC(kernel='linear', C=1, random_state=42),
}

REGRESSION_MODELS = {
    "rf": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "gb": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "ridge": lambda: Ridge(),
    "lasso": lambda: Lasso(),
    "dt": lambda: DecisionTreeRegressor(random_state=42),
    "knn": lambda: KNeighborsRegressor(),
    "svr": lambda: SVR(kernel='linear', C=1),
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


ALGORITHM = {
    "type": "string",
    "description": "Algorithm key of a trained model (see train_models).",
}


@register_tool(
    "train_test_split",
    "Practice split: divide the training rows into a part to learn from and a part to check against. "
    "The locked test rows are not used.",
    {
        "test_size": {"type": "number", "description": "Fraction for the test set (default 0.2)."},
        "stratify": {"type": "boolean", "description": "Stratify by target (classification only)."},
    },
)
def do_train_test_split(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Split data into training and test sets."""
    if not session.target_column:
        return {"success": False, "message": "Select a target column first (use 'select_target' tool)."}

    validate_column(session, session.target_column)

    test_size = args.get("test_size", 0.2)
    stratify = args.get("stratify", False)

    X = session.df.drop(columns=[session.target_column])
    y = session.df[session.target_column]

    # Handle non-numeric columns: warn and auto-encode
    object_cols = categorical_columns(X)
    if object_cols:
        X = pd.get_dummies(X, columns=object_cols, drop_first=True)

    # Fill remaining NaN
    X = X.fillna(0)

    stratify_param = y if stratify and session.problem_type == "classification" else None

    session.X_train, session.X_test, session.y_train, session.y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_param
    )

    # TODO(piece 3): evaluate on the locked test rows instead of this practice split
    msg = (
        f"Practice split of the training rows: {len(session.X_train)} to learn from, "
        f"{len(session.X_test)} to check against (test_size={test_size}). "
        f"The {len(session.test_df) if session.is_split else 0} locked test rows stay untouched."
    )
    if object_cols:
        msg += f" Auto-encoded {len(object_cols)} categorical columns: {object_cols}."

    return {"success": True, "message": msg}


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
)
def train_models(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Train one or more ML models."""
    if session.X_train is None:
        return {"success": False, "message": "Split the data first (use 'train_test_split' tool)."}

    alg_names = args.get("algorithms", ["rf", "gb"])
    cv_folds = args.get("cv_folds", 5)

    model_registry = CLASSIFICATION_MODELS if session.problem_type == "classification" else REGRESSION_MODELS
    results = []

    for alg in alg_names:
        if alg not in model_registry:
            available = ", ".join(model_registry.keys())
            results.append({"model": alg, "error": f"Unknown algorithm. Available: {available}"})
            continue

        model = model_registry[alg]()
        model.fit(session.X_train, session.y_train)
        y_pred = model.predict(session.X_test)

        if session.problem_type == "regression":
            score = round(float(r2_score(session.y_test, y_pred)), 4)
            mae = round(float(mean_absolute_error(session.y_test, y_pred)), 4)
            rmse = round(float(np.sqrt(mean_squared_error(session.y_test, y_pred))), 4)
            result = {"model": alg, "r2_score": score, "mae": mae, "rmse": rmse}
        else:
            acc = round(float(accuracy_score(session.y_test, y_pred)), 4)
            f1 = round(float(f1_score(session.y_test, y_pred, average='weighted', zero_division=0)), 4)
            report = classification_report(session.y_test, y_pred, zero_division=0)
            result = {"model": alg, "accuracy": acc, "f1_score": f1, "report": report}

        # Cross-validation
        cv_scores = cross_val_score(model, session.X_train, session.y_train, cv=min(cv_folds, len(session.X_train)))
        result["cv_mean"] = round(float(cv_scores.mean()), 4)
        result["cv_std"] = round(float(cv_scores.std()), 4)

        session.trained_models[alg] = model
        results.append(result)

    # Find best model
    score_key = "cv_mean"
    valid_results = [r for r in results if score_key in r]
    best = max(valid_results, key=lambda r: r[score_key])["model"] if valid_results else None

    return {
        "success": True,
        "message": f"Trained {len(alg_names)} models. Best: {best}.",
        "evidence": {"results": results, "best_model": best},
    }


@register_tool("auto_ml", "Grid-search several algorithms with hyperparameter tuning and report the best.")
def auto_ml(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Automatically find the best model with hyperparameter tuning (AutoML)."""
    if session.X_train is None:
        return {"success": False, "message": "Split the data first (use 'train_test_split' tool)."}

    grids = AUTOML_CLASSIFICATION_GRIDS if session.problem_type == "classification" else AUTOML_REGRESSION_GRIDS
    scoring = "accuracy" if session.problem_type == "classification" else "r2"

    results = []
    best_score = -float('inf')
    best_model_name = None

    for name, config in grids.items():
        try:
            grid_search = GridSearchCV(
                config["model"], config["params"],
                cv=3, scoring=scoring, n_jobs=-1, error_score='raise'
            )
            grid_search.fit(session.X_train, session.y_train)

            best_estimator = grid_search.best_estimator_
            y_pred = best_estimator.predict(session.X_test)

            if session.problem_type == "regression":
                score = round(float(r2_score(session.y_test, y_pred)), 4)
                result = {"model": name, "best_params": grid_search.best_params_,
                         "r2_score": score, "cv_best_score": round(float(grid_search.best_score_), 4)}
            else:
                acc = round(float(accuracy_score(session.y_test, y_pred)), 4)
                result = {"model": name, "best_params": grid_search.best_params_,
                         "accuracy": acc, "cv_best_score": round(float(grid_search.best_score_), 4)}

            session.trained_models[name] = best_estimator
            results.append(result)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model_name = name

        except Exception as e:
            results.append({"model": name, "error": str(e)})

    return {
        "success": True,
        "message": f"AutoML complete. Best model: {best_model_name} (CV score: {best_score:.4f}).",
        "evidence": {"results": results, "best_model": best_model_name},
    }


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

    model_name = args.get("algorithm", list(session.trained_models.keys())[0])
    if model_name not in session.trained_models:
        return {"success": False, "message": f"Model '{model_name}' not found. Available: {list(session.trained_models.keys())}"}

    model = session.trained_models[model_name]
    features = args.get("features", {})

    if not features:
        return {"success": False, "message": "Provide 'features' dict with column values."}

    input_data = pd.DataFrame([features])

    # Apply scaling if exists
    if session.scaler:
        try:
            cols = input_data.columns.tolist()
            input_data[cols] = session.scaler.transform(input_data[cols])
        except Exception:
            pass  # If columns don't match scaler, predict without scaling

    prediction = model.predict(input_data)[0]

    # Decode if target was label-encoded
    if session.target_column in session.label_encoders:
        try:
            prediction = session.label_encoders[session.target_column].inverse_transform([int(prediction)])[0]
        except Exception:
            pass

    return {
        "success": True,
        "message": f"Prediction: {prediction} (using {model_name}).",
        "evidence": {"prediction": str(prediction), "model_used": model_name, "input_features": features},
    }


@register_tool("explain_model", "Show which features drive a trained model's predictions.", {"algorithm": ALGORITHM})
def explain_model(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate SHAP feature importance explanations."""
    if not session.trained_models:
        return {"success": False, "message": "Train models first."}

    model_name = args.get("algorithm", list(session.trained_models.keys())[0])
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


@register_tool("save_model", "Save a trained model (with its preprocessing) to disk.", {"algorithm": ALGORITHM})
def save_model(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Save a trained model to disk."""
    if not session.trained_models:
        return {"success": False, "message": "No trained models to save."}

    model_name = args.get("algorithm", list(session.trained_models.keys())[0])
    if model_name not in session.trained_models:
        return {"success": False, "message": f"Model '{model_name}' not found."}

    save_dir = os.path.join(settings.DATA_DIR, "models")
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"model_{model_name}_{session.session_id[:8]}.pkl")
    with open(path, 'wb') as f:
        pickle.dump({
            'model': session.trained_models[model_name],
            'scaler': session.scaler,
            'encoders': session.label_encoders,
            'target': session.target_column,
            'problem_type': session.problem_type,
            'feature_names': session.X_train.columns.tolist() if session.X_train is not None else [],
        }, f)

    return {
        "success": True,
        "message": f"Model '{model_name}' saved to {path}.",
    }
