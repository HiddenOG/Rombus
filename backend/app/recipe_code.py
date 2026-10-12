"""Readable Python for every recipe step, in the same learn/apply shape the engine uses.

The code shown is equivalent to what app/recipe.py runs, written plainly with the step's
actual settings, so a user can read it, copy it, or run it in their own notebook.
"""

import json
from textwrap import dedent, indent
from typing import Any, Callable, Dict, Optional

import pandas as pd

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
    text = df[{col!r}].astype(str)
    # A range like "2100 - 2850" becomes its midpoint
    ends = text.str.extract(r"^\\s*(\\d[\\d,]*\\.?\\d*)\\s*-\\s*(\\d[\\d,]*\\.?\\d*)\\s*$")
    low = pd.to_numeric(ends[0].str.replace(",", ""), errors="coerce")
    high = pd.to_numeric(ends[1].str.replace(",", ""), errors="coerce")
    # Anything else: keep only digits, dots and minus signs ("$1,200" -> 1200, "2 BHK" -> 2)
    plain = pd.to_numeric(text.str.replace(r"[^\\d.\\-]", "", regex=True), errors="coerce").where(df[{col!r}].notna())
    is_range = low.notna() & high.notna()
    df[{col!r}] = ((low + high) / 2).where(is_range, plain) if is_range.any() else plain
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


@builder("drop_missing_rows")
def _drop_missing_rows(args, target):
    cols = args.get("columns") or []
    subset = f"subset={cols!r}" if cols else ""
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    # TRAINING ROWS ONLY: you still have to predict for test rows and new customers with blanks
    return df.dropna({subset}).reset_index(drop=True)
''')


@builder("drop_rows_where")
def _drop_rows_where(args, target):
    from app.expressions import filter_code
    conditions = args.get("conditions", "")
    columns = args.get("_columns")  # filled in by step_code: the columns this step sees
    mask = filter_code(columns, conditions, frame="df") if columns is not None else f"<conditions: {conditions}>"
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    # TRAINING ROWS ONLY: you still have to predict for test rows and new customers like these
    drop = {mask}  # {conditions}
    return df[~drop].reset_index(drop=True)
''')


@builder("change_type")
def _change_type(args, target):
    col, to = args.get("column"), args.get("to")
    line = {
        "text": f"df[{col!r}] = df[{col!r}].astype(str).where(df[{col!r}].notna())  # keep blanks blank",
        "whole number": f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce').round().astype('Int64')",
    }.get(to, f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce')  # unconvertible values become blanks")
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    {line}
    return df
''')


@builder("rename_column")
def _rename(args, target):
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    return df.rename(columns={{{args.get("column")!r}: {(args.get("new_name") or "").strip()!r}}})
''')


@builder("fill_value")
def _fill_value(args, target):
    col, value = args.get("column"), args.get("value")
    columns = args.get("_columns")
    if columns is not None and col in columns and pd.api.types.is_numeric_dtype(columns[col]):
        value = int(float(value)) if pd.api.types.is_integer_dtype(columns[col]) else float(value)
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    df[{col!r}] = df[{col!r}].fillna({value!r})  # the same value everywhere: nothing to learn
    return df
''')


@builder("add_column")
def _add_column(args, target):
    from app.expressions import formula_code
    name, formula = (args.get("name") or "").strip(), args.get("formula", "")
    columns = args.get("_columns")
    expression = formula_code(formula, columns.columns, frame="df") if columns is not None else f"<formula: {formula}>"
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    # {formula}   (row by row: the same on training rows, test rows and new customers)
    df[{name!r}] = ({expression}).replace([np.inf, -np.inf], np.nan)
    return df
''')


@builder("group_rare_categories")
def _group_rare(args, target):
    col, max_rows, label = args.get("column"), int(args.get("max_rows") or 10), args.get("label") or "other"
    return _code(f'''
def learn(train):
    # Which values are common enough to keep: counted on the TRAINING rows only
    counts = train[{col!r}].dropna().astype(str).str.strip().value_counts()
    return {{"keep": sorted(counts[counts > {max_rows}].index)}}


def apply(df, learned):
    values = df[{col!r}].astype(str).str.strip().where(df[{col!r}].notna())
    # Rare values, and values never seen in training, become {label!r}
    df[{col!r}] = values.where(values.isin(learned["keep"]) | values.isna(), {label!r})
    return df
''')


@builder("remove_outliers")
def _remove_outliers(args, target):
    from app.expressions import formula_code
    columns = args.get("_columns")
    value = args.get("value", "")
    value_code = formula_code(value, columns.columns, frame="df") if columns is not None else f"<value: {value}>"
    method = args.get("method") or "std"
    k = float(args["threshold"]) if args.get("threshold") not in (None, "") else (3.0 if method == "std" else 1.5)
    group = args.get("group_by")
    groups = f"df[{group!r}].astype(str)" if group else 'pd.Series("all", index=df.index)'
    if method == "std":
        bounds = (f"center, spread = grouped.transform('mean'), grouped.transform('std')\n"
                  f"    low, high = center - {k} * spread, center + {k} * spread")
    else:
        bounds = (f"q1, q3 = grouped.transform(lambda v: v.quantile(0.25)), grouped.transform(lambda v: v.quantile(0.75))\n"
                  f"    low, high = q1 - {k} * (q3 - q1), q3 + {k} * (q3 - q1)")
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    # TRAINING ROWS ONLY: you still have to predict for test rows and new customers
    value = ({value_code}).replace([np.inf, -np.inf], np.nan)  # {value}
    grouped = value.groupby({groups})
    {bounds}
    keep = value.isna() | low.isna() | high.isna() | value.between(low, high)
    return df[keep].reset_index(drop=True)
''')


def _fill_expr(col: str, strategy: str, whole: bool = False) -> str:
    expr = {
        "mean": f"train[{col!r}].mean()",
        "median": f"train[{col!r}].median()",
        "mode": f"train[{col!r}].mode()[0]",
    }[strategy]
    # A whole-number column gets a whole number (1.88 rooms -> 2), as in the app
    return f"int(round({expr}))" if whole and strategy != "mode" else expr


@builder("impute_missing")
def _impute_missing(args, target):
    col, strategy = args.get("column"), args.get("strategy", "mean")
    columns = args.get("_columns")
    whole = columns is not None and col in columns and pd.api.types.is_integer_dtype(columns[col])
    return _code(f'''
def learn(train):
    # The {strategy} of {col!r}, from the TRAINING rows only
    return {{"fill": {{{col!r}: {_fill_expr(col, strategy, whole)}}}}}


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
            if pd.api.types.is_integer_dtype(train[col]) and {(strategy or "mean")!r} != "mode":
                fill[col] = int(round(fill[col]))  # whole-number columns get a whole number
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


@builder("custom_code")
def _custom(args, target):
    return args.get("code", "")


def step_code(line: Line, target: Optional[str], with_learned: bool = True) -> str:
    """The step as Python, followed by what it actually learned from this session's training rows."""
    build = BUILDERS.get(line.tool)
    if build is None:
        return f"# No code view for {line.tool}"
    if line.tool == "custom_code":
        code = build(line.args, target).rstrip()
    else:
        # `_columns`: what the step saw (no rows), so the code uses the right columns and value types
        code = "import numpy as np\nimport pandas as pd\n\n\n" + build({**line.args, "_columns": line.schema}, target)
    if not with_learned:
        return code
    if line.learned:
        learned = json.dumps(line.learned, indent=2, default=str)
        code += "\n\n\n# What learn() returned for these training rows:\n" + "\n".join(f"# {l}" for l in learned.splitlines())
    return code
