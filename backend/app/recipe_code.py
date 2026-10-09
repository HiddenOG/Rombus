"""Readable Python for every recipe step, in the same learn/apply shape the engine uses.

The code shown is equivalent to what app/recipe.py runs, written plainly with the step's
actual settings, so a user can read it, copy it, or run it in their own notebook.
"""

import json
from textwrap import dedent, indent
from typing import Any, Callable, Dict, Optional

from app.recipe import Line

Builder = Callable[[Dict[str, Any], Optional[str]], str]
BUILDERS: Dict[str, Builder] = {}


def builder(name: str):
    def decorator(fn: Builder) -> Builder:
        BUILDERS[name] = fn
        return fn
    return decorator


def _code(text: str) -> str:
    return dedent(text).strip("\n")


def _features(target: Optional[str]) -> str:
    return f"train.columns.drop({target!r})" if target else "train.columns"


NOTHING_TO_LEARN = '''
def learn(train):
    return {}  # nothing to learn: this step does the same thing every time
'''


@builder("remove_columns")
def _remove_columns(args, target):
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    return df.drop(columns={args.get("columns", [])!r}, errors="ignore")
''')


@builder("clean_numeric_string")
def _clean_numeric(args, target):
    col = args.get("column")
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    # Keep only digits, dots and minus signs ("$1,200" -> "1200"), then convert to numbers
    text = df[{col!r}].astype(str).str.replace(r"[^\\d.\\-]", "", regex=True)
    df[{col!r}] = pd.to_numeric(text, errors="coerce")
    return df
''')


@builder("standardize_text")
def _standardize_text(args, target):
    col = args.get("column")
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    df[{col!r}] = df[{col!r}].str.lower().str.strip()
    return df
''')


@builder("log_transform")
def _log_transform(args, target):
    col = args.get("column")
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    df[{col!r}] = np.log1p(df[{col!r}].clip(lower=0))  # log(1 + x) squeezes long tails
    return df
''')


@builder("remove_duplicates")
def _remove_duplicates(args, target):
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code('''
def apply(df, learned):
    # TRAINING ROWS ONLY: never drop rows from the test set or from customers we must predict for
    return df.drop_duplicates().reset_index(drop=True)
''')


def _fill_expr(col: str, strategy: str) -> str:
    return {
        "mean": f"train[{col!r}].mean()",
        "median": f"train[{col!r}].median()",
        "mode": f"train[{col!r}].mode()[0]",
    }[strategy]


@builder("impute_missing")
def _impute_missing(args, target):
    col, strategy = args.get("column"), args.get("strategy", "mean")
    return _code(f'''
def learn(train):
    # The {strategy} of {col!r}, from the TRAINING rows only
    return {{"fill": {{{col!r}: {_fill_expr(col, strategy)}}}}}


def apply(df, learned):
    return df.fillna(learned["fill"])  # same value for training rows, test rows and new customers
''')


@builder("impute_all_missing")
def _impute_all(args, target):
    strategy = args.get("strategy")
    numeric = {"mean": "train[col].mean()", "median": "train[col].median()", "mode": "train[col].mode()[0]"}
    numeric_expr = numeric.get(strategy or "mean")
    return _code(f'''
def learn(train):
    fill = {{}}
    for col in {_features(target)}:
        if pd.api.types.is_numeric_dtype(train[col]):
            fill[col] = {numeric_expr}
        else:
            fill[col] = train[col].mode()[0]  # most common value for text columns
    return {{"fill": fill}}


def apply(df, learned):
    return df.fillna(learned["fill"])
''')


ENCODE_APPLY = '''
def apply(df, learned):
    for col, categories in learned["classes"].items():
        if col not in df.columns:
            continue
{body}
    return df
'''

LABEL_BODY = '''\
codes = {c: i for i, c in enumerate(categories)}
# Categories never seen in training get -1
df[col] = df[col].astype(str).map(codes).where(df[col].notna()).fillna(-1).astype(int)'''

ONEHOT_BODY = '''\
for c in categories:
    df[f"{col}_{c}"] = (df[col].astype(str) == c).astype(int)  # unseen category -> all zeros
df = df.drop(columns=[col])'''


def _encode_apply(method: str) -> str:
    body = ONEHOT_BODY if method == "onehot" else LABEL_BODY
    return _code(ENCODE_APPLY.format(body=indent(body, " " * 8)))


@builder("encode_categorical")
def _encode_one(args, target):
    col, method = args.get("column"), args.get("method", "label")
    return _code(f'''
def learn(train):
    # Every category that appears in the TRAINING rows
    return {{"classes": {{{col!r}: sorted(train[{col!r}].dropna().astype(str).unique())}}}}
''') + "\n\n\n" + _encode_apply(method)


@builder("encode_all_categorical")
def _encode_all(args, target):
    method = args.get("method", "label")
    return _code(f'''
def learn(train):
    text_columns = [c for c in {_features(target)}
                    if not pd.api.types.is_numeric_dtype(train[c])]
    return {{"classes": {{c: sorted(train[c].dropna().astype(str).unique()) for c in text_columns}}}}
''') + "\n\n\n" + _encode_apply(method)


SCALE_APPLY = '''
def apply(df, learned):
    for col, s in learned["scale"].items():
        df[col] = (df[col] - s["mean"]) / s["std"]  # training average and spread, never the test rows
    return df
'''


@builder("scale_features")
def _scale(args, target):
    cols = args.get("columns")
    choose = f"{cols!r}" if cols else f"[c for c in {_features(target)} if pd.api.types.is_numeric_dtype(train[c])]"
    return _code(f'''
def learn(train):
    columns = {choose}
    return {{"scale": {{c: {{"mean": train[c].mean(), "std": train[c].std(ddof=0) or 1.0}} for c in columns}}}}
''') + "\n\n\n" + _code(SCALE_APPLY)


@builder("scale_all_features")
def _scale_all(args, target):
    return _scale({}, target)


def step_code(line: Line, target: Optional[str]) -> str:
    """The step as Python, followed by what it actually learned from this session's training rows."""
    build = BUILDERS.get(line.tool)
    if build is None:
        return f"# No code view for {line.tool}"
    code = "import numpy as np\nimport pandas as pd\n\n\n" + build(line.args, target)
    if line.learned:
        learned = json.dumps(line.learned, indent=2, default=str)
        code += "\n\n\n# What learn() returned for these training rows:\n" + "\n".join(f"# {l}" for l in learned.splitlines())
    return code
