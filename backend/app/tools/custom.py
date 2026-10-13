"""Custom views and custom models: user-written code for "looking" and "training" cells.

The referee cells stay protected on purpose: choosing the target (the lock), the final exam and
predict always run the app's own code, so their guarantees ("the test rows were never seen",
"this score is honest") can't be edited away.
"""

from typing import Any, Dict

from app.session import DataSession
from app.tools import register_tool
from app import sandbox

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
    # In the sandbox, on a copy of the training rows (also called df: the name pandas habit reaches for)
    evidence = sandbox.run({"kind": "view", "code": args.get("code") or "", "frames": {"train": session.df}})["evidence"]
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
    from app.tools.model import ensure_practice_split, finish_training, safe_cv_folds

    auto_split = ensure_practice_split(session)
    # Built, trained and scored in the sandbox (exactly like fit_and_score does for built-in models);
    # the trained models come back as skops files, loaded here only if they're scikit-learn parts
    done = sandbox.run({
        "kind": "models", "code": args.get("code") or "", "problem_type": session.problem_type,
        "cv": {"time": bool(session.time_column), "folds": safe_cv_folds(session, 5)},
        "frames": {"X_learn": session.X_train, "y_learn": session.y_train.to_frame(),
                   "X_check": session.X_test, "y_check": session.y_test.to_frame()},
    })
    results = []
    for result in done["results"]:
        name = result["model"]
        if name in done["fitted"]:
            try:
                session.trained_models[name] = sandbox.load_model(done["fitted"][name])
                session.model_results[name] = result
            except ValueError as e:
                result = {"model": name, "error": str(e)}
        results.append(result)

    best = finish_training(session, results)
    msg = f"Trained {len(results)} custom model{'s' if len(results) != 1 else ''}. Best on the practice split: {best}."
    if auto_split:
        msg = f"{auto_split} {msg}"
    if done.get("output"):
        msg += f"\n{done['output'].strip()}"
    return {"success": True, "message": msg, "evidence": {"results": results, "best_model": best}}
