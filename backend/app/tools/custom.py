"""Custom views and custom models: user-written code for "looking" and "training" cells.

The referee cells stay protected on purpose: choosing the target (the lock), the final exam and
predict always run the app's own code, so their guarantees ("the test rows were never seen",
"this score is honest") can't be edited away.
"""

from typing import Any, Dict

from app.session import DataSession
from app.tools import register_tool
from app.user_code import result_evidence, run_user_code

VIEW_TEMPLATE = '''import numpy as np
import pandas as pd

# `train` = a copy of the training rows after your recipe steps (changing it changes nothing).
# Set `result` to a table, a number or text to show it. print() output and matplotlib charts are shown too.

result = train.describe()
'''

MODELS_TEMPLATE = '''from sklearn.ensemble import RandomForestClassifier

# Define the models to compare as {name: model}. Any scikit-learn-style model works.
# The app trains each on the practice split and scores it exactly like the built-in models.
# You can look at X_learn / y_learn (the practice training part) while deciding.

models = {
    "my_forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
}
'''

CODE_SPEC = {"type": "string", "format": "code", "description": "Python code."}


@register_tool(
    "custom_view",
    "Your own code to look at the training rows: set `result` to a table, number or text.",
    {"code": {**CODE_SPEC, "default": VIEW_TEMPLATE}},
    ["code"],
    kind="view",
    ai=False,
)
def custom_view(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    run = run_user_code(args.get("code") or "", {"train": session.df.copy()})
    evidence = result_evidence(run.namespace.get("result"))
    if run.output:
        evidence["output"] = (evidence.get("output", "") + "\n" + run.output).strip()
    if run.images:
        evidence["images"] = run.images
    if not evidence:
        return {"success": True, "message": "Your code ran but showed nothing. Set `result`, print(), or draw a chart."}
    return {"success": True, "message": "Custom view ran on the training rows.", "evidence": evidence}


@register_tool(
    "custom_models",
    "Your own models to train and compare: define `models = {name: model}`.",
    {"code": {**CODE_SPEC, "default": MODELS_TEMPLATE}},
    ["code"],
    kind="action",
    ai=False,
)
def custom_models(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    from app.tools.model import ensure_practice_split, finish_training, fit_and_score, safe_cv_folds

    auto_split = ensure_practice_split(session)
    run = run_user_code(args.get("code") or "", {
        "X_learn": session.X_train.copy(), "y_learn": session.y_train.copy(),
    })
    models = run.namespace.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Your code must define `models` as a dict like {\"name\": SomeModel()}.")
    for name, model in models.items():
        if not (hasattr(model, "fit") and hasattr(model, "predict")):
            raise ValueError(f"'{name}' isn't a model: it needs fit() and predict() methods.")

    cv_folds = safe_cv_folds(session, 5)
    results = []
    for name, model in models.items():
        try:
            results.append(fit_and_score(session, str(name), model, cv_folds))
        except Exception as e:
            results.append({"model": str(name), "error": f"{type(e).__name__}: {e}"})

    best = finish_training(session, results)
    msg = f"Trained {len(models)} custom model{'s' if len(models) != 1 else ''}. Best on the practice split: {best}."
    if auto_split:
        msg = f"{auto_split} {msg}"
    if run.output:
        msg += f"\n{run.output.strip()}"
    return {"success": True, "message": msg, "evidence": {"results": results, "best_model": best}}
