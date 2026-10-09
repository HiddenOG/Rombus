"""Readable Python for actions and views (everything that isn't a recipe step).

Recipe steps show the exact code that runs (see recipe_code.py). For actions and views this
module writes the equivalent pandas / scikit-learn code with the notebook's real settings, so a
user can see what happened under the hood and reproduce it in their own notebook.
"""

from textwrap import dedent
from typing import Any, Callable, Dict

import numpy as np

from app.session import DataSession

Builder = Callable[[DataSession, Dict[str, Any]], str]
BUILDERS: Dict[str, Builder] = {}

HEADER = "import numpy as np\nimport pandas as pd\n"
TRAIN_NOTE = "# `train` = the training rows after your recipe steps (the locked test rows are never used here)\n"


def builder(*names: str):
    def decorator(fn: Builder) -> Builder:
        for name in names:
            BUILDERS[name] = fn
        return fn
    return decorator


def _code(text: str) -> str:
    return dedent(text).strip("\n")


def _stratifies(session: DataSession, test_size: float) -> bool:
    """Same rule the app uses when locking: stratify if every class can appear in both piles."""
    if session.problem_type != "classification" or not session.target_column:
        return False
    y = session.original_df[session.target_column].dropna()
    counts = y.value_counts()
    return bool(counts.min() >= 2 and int(np.ceil(len(y) * test_size)) >= len(counts))


@builder("select_target")
def _select_target(session, args):
    target = args.get("column") or session.target_column
    size = float(args.get("test_size") or session.test_size or 0.2)
    if size >= 1:
        size /= 100
    stratify = f"df[{target!r}]" if _stratifies(session, size) else "None"
    return HEADER + "from sklearn.model_selection import train_test_split\n\n" + _code(f'''
df = pd.read_csv({session.filename!r})
df = df.dropna(subset=[{target!r}])  # rows with no answer can't be learned from or graded

# Lock away {size:.0%} of the rows as the final exam. Nothing looks at them until final_exam.
train, test = train_test_split(
    df,
    test_size={size},
    random_state=42,
    stratify={stratify},  # stratify keeps the same mix of answers in both piles
)
''')


# Views end by setting `result`, so the code shown is directly runnable as a custom view.
VIEW_NOTE = "# `result` is what gets shown. Uncomment a .plot line to draw a chart (needs matplotlib).\n"


@builder("preview_data")
def _preview(session, args):
    rows = int(args.get("rows") or 5)
    return HEADER + TRAIN_NOTE + VIEW_NOTE + f"\nresult = train.head({rows})"


@builder("get_data_info")
def _info(session, args):
    return HEADER + TRAIN_NOTE + VIEW_NOTE + "\ntrain.info()  # column types and non-blank counts\nresult = train.describe(include='all')"


@builder("check_duplicates")
def _duplicates(session, args):
    return HEADER + TRAIN_NOTE + VIEW_NOTE + "\nresult = f\"{train.duplicated().sum()} duplicate rows out of {len(train)}\""


@builder("plot_histogram")
def _histogram(session, args):
    col = args.get("column")
    return HEADER + TRAIN_NOTE + VIEW_NOTE + "\n" + _code(f'''
values = train[{col!r}].dropna()
if pd.api.types.is_numeric_dtype(values):
    bins = min(30, max(10, len(values) // 10))
    counts, edges = np.histogram(values, bins=bins)  # how many rows fall in each range
    result = pd.DataFrame({{"from": edges[:-1].round(2), "to": edges[1:].round(2), "rows": counts}})
    # values.plot.hist(bins=bins)
else:
    result = values.value_counts().head(20)  # text columns: count each category
    # result.plot.bar()
''')


@builder("plot_boxplot")
def _boxplot(session, args):
    col = args.get("column")
    return HEADER + TRAIN_NOTE + VIEW_NOTE + "\n" + _code(f'''
values = train[{col!r}].dropna()
q1, median, q3 = values.quantile(0.25), values.median(), values.quantile(0.75)
iqr = q3 - q1
outliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]  # beyond the whiskers
result = pd.Series({{"min": values.min(), "q1": q1, "median": median, "q3": q3, "max": values.max(),
                    "outliers": len(outliers)}})
# values.plot.box()
''')


@builder("plot_correlation")
def _correlation(session, args):
    return HEADER + TRAIN_NOTE + VIEW_NOTE + "\n" + _code('''
result = train.select_dtypes(include="number").corr().round(2)  # -1 opposite … 0 unrelated … +1 together
''')


@builder("plot_scatterplot")
def _scatter(session, args):
    x, y, hue = args.get("x"), args.get("y"), args.get("hue")
    cols = list(dict.fromkeys(c for c in (x, y, hue) if c))
    return HEADER + TRAIN_NOTE + VIEW_NOTE + "\n" + _code(f'''
result = train[{cols!r}].dropna()
if len(result) > 500:
    result = result.sample(500, random_state=42)  # keep the chart fast
# result.plot.scatter(x={x!r}, y={y!r})
''')


def _features_block(target: str) -> str:
    return _code(f'''
def model_features(df):
    # All-numeric table for the model: leftover text columns are one-hot encoded, blanks become 0
    X = df.drop(columns=[{target!r}], errors="ignore")
    text = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return pd.get_dummies(X, columns=text, dtype=int).fillna(0)
''')


@builder("train_test_split")
def _practice_split(session, args):
    target = session.target_column
    size = float(args.get("test_size") or 0.2)
    stratify = "y" if args.get("stratify") and session.problem_type == "classification" else "None"
    return (HEADER + "from sklearn.model_selection import train_test_split\n\n" + TRAIN_NOTE + "\n"
            + _features_block(target) + "\n\n\n" + _code(f'''
X, y = model_features(train), train[{target!r}]

# A PRACTICE split of the training rows, used to compare models. The locked test rows stay untouched.
X_learn, X_check, y_learn, y_check = train_test_split(X, y, test_size={size}, random_state=42, stratify={stratify})
'''))


def _model_lines(session, algorithms) -> str:
    from app.tools.model import CLASSIFICATION_MODELS, REGRESSION_MODELS
    registry = CLASSIFICATION_MODELS if session.problem_type == "classification" else REGRESSION_MODELS
    lines = [f"    {a!r}: {registry[a]()!r}," for a in algorithms if a in registry]
    return "{\n" + "\n".join(lines) + "\n}"


def _sklearn_imports(session, algorithms) -> str:
    from app.tools.model import CLASSIFICATION_MODELS, REGRESSION_MODELS
    registry = CLASSIFICATION_MODELS if session.problem_type == "classification" else REGRESSION_MODELS
    imports = {}
    for a in algorithms:
        if a in registry:
            cls = type(registry[a]())
            imports.setdefault(cls.__module__.rsplit("._", 1)[0], set()).add(cls.__name__)
    return "\n".join(f"from {mod} import {', '.join(sorted(names))}" for mod, names in sorted(imports.items()))


@builder("train_models")
def _train(session, args):
    algorithms = args.get("algorithms") or ["rf", "gb"]
    score = "accuracy_score" if session.problem_type == "classification" else "r2_score"
    return (HEADER + _sklearn_imports(session, algorithms) + "\n"
            + f"from sklearn.metrics import {score}\nfrom sklearn.model_selection import cross_val_score\n\n"
            + "# X_learn / X_check / y_learn / y_check come from the practice split (see train_test_split)\n\n"
            + _code(f'''
models = {_model_lines(session, algorithms)}

for name, model in models.items():
    model.fit(X_learn, y_learn)
    practice = {score}(y_check, model.predict(X_check))
    cv = cross_val_score(model, X_learn, y_learn, cv={int(args.get("cv_folds") or 5)})  # several mini-exams
    print(name, round(practice, 4), round(cv.mean(), 4))
'''))


@builder("auto_ml")
def _automl(session, args):
    scoring = "accuracy" if session.problem_type == "classification" else "r2"
    return HEADER + "from sklearn.model_selection import GridSearchCV\n\n" + _code(f'''
# For each algorithm, try every combination of settings with 3-fold cross-validation
# on the practice-split training part, and keep the best combination.
search = GridSearchCV(model, param_grid, cv=3, scoring={scoring!r})
search.fit(X_learn, y_learn)
best_model, best_settings = search.best_estimator_, search.best_params_
''')


@builder("final_exam")
def _exam(session, args):
    target = session.target_column
    is_clf = session.problem_type == "classification"
    metric = "accuracy_score" if is_clf else "r2_score"
    baseline = (f"y_test.eq(train[{target!r}].mode()[0]).mean()  # always guess the most common answer"
                if is_clf else
                f"r2_score(y_test, [train[{target!r}].mean()] * len(y_test))  # always guess the average")
    return (HEADER + "from sklearn.base import clone\n" + f"from sklearn.metrics import {metric}\n\n"
            + _features_block(target) + "\n\n\n" + _code(f'''
# 1. Retrain the chosen model on ALL training rows (the practice split was only for choosing)
X_all = model_features(train)
final_model = clone(chosen_model).fit(X_all, train[{target!r}])

# 2. Prepare the locked test rows with the recipe's TRAINING sticky notes: apply() only, never learn()
prepared = test.copy()
for step in recipe:
    if not step.train_only:
        prepared = step.apply(prepared, step.learned)

X_test = model_features(prepared).reindex(columns=X_all.columns, fill_value=0)
y_test = prepared[{target!r}]

# 3. Score once, and compare with just guessing
score = {metric}(y_test, final_model.predict(X_test))
baseline = {baseline}
'''))


@builder("predict")
def _predict(session, args):
    features = args.get("features") or {}
    return HEADER + "\n" + _features_block(session.target_column) + "\n\n\n" + _code(f'''
new_row = pd.DataFrame([{features!r}])

# The new case goes through the recipe with the TRAINING sticky notes, like the test rows did
for step in recipe:
    if not step.train_only:
        new_row = step.apply(new_row, step.learned)

X = model_features(new_row).reindex(columns=X_all.columns, fill_value=0)  # same columns as training
prediction = model.predict(X)[0]
confidence = model.predict_proba(X)[0]  # how sure it is about each answer (classifiers)
''')


@builder("explain_model")
def _explain(session, args):
    return HEADER + "from sklearn.inspection import permutation_importance\n\n" + _code('''
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_  # tree models measure this while learning
else:
    # Shuffle one column at a time and see how much the practice score drops
    importance = permutation_importance(model, X_check, y_check, n_repeats=10, random_state=42).importances_mean
''')


@builder("save_model")
def _save(session, args):
    return "import pickle\n\n" + _code('''
# The model is saved together with the recipe (steps + sticky notes), so whoever loads it
# can prepare new rows exactly like the training rows before predicting.
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "recipe": recipe, "target": target}, f)
''')


@builder("undo")
def _undo(session, args):
    return _code('''
# Undo = cross out the last recipe step, then rebuild the training rows from scratch:
recipe = recipe[:-1]
train = raw_train.copy()
for step in recipe:
    learned = step.learn(train)
    train = step.apply(train, learned)
''')


@builder("transition_stage")
def _stage(session, args):
    return "# Moves the notebook to the next ASEMIC stage. No data changes, so there's no data code to show."


def action_code(session: DataSession, tool: str, args: Dict[str, Any]) -> str:
    build = BUILDERS.get(tool)
    if build is None:
        return f"# No code view for '{tool}' yet."
    return build(session, args or {})


VIEW_TOOLS = {"preview_data", "get_data_info", "check_duplicates", "plot_histogram", "plot_boxplot",
              "plot_correlation", "plot_scatterplot"}

# The referee: their code is shown but can't be edited, or their guarantees would mean nothing
PROTECTED = {
    "select_target": "This is the lock. If its code could be edited, you could choose which rows end up in the exam.",
    "final_exam": "This is the referee. If its code could be edited, you could change how the honest score is calculated.",
    "predict": "Predictions always go through the recipe and model exactly as trained, so they match the exam.",
}


def models_template(session: DataSession, algorithms) -> str:
    """Editable starting point for custom models: the cell's own models, written out."""
    algorithms = algorithms or ["rf", "gb"]
    return (_sklearn_imports(session, algorithms) + "\n\n"
            + "# Define the models to compare as {name: model}. Any scikit-learn-style model works.\n"
            + "# Change settings, add models, or remove some. The app trains and scores them like built-ins.\n\n"
            + f"models = {_model_lines(session, algorithms)}\n")


def code_info(session: DataSession, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """The code for a cell, and whether / how it can be edited."""
    args = args or {}
    if tool in ("custom_view", "custom_models"):
        code = args.get("code", "")
        return {"code": code, "custom_tool": tool, "template": code, "protected": None}
    code = action_code(session, tool, args)
    if tool in VIEW_TOOLS:
        return {"code": code, "custom_tool": "custom_view", "template": code, "protected": None}
    if tool in ("train_models", "auto_ml"):
        return {"code": code, "custom_tool": "custom_models", "template": models_template(session, args.get("algorithms")),
                "protected": None}
    return {"code": code, "custom_tool": None, "template": None, "protected": PROTECTED.get(tool)}
