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
    # Anything else: the first number in the text ("$1,200" -> 1200, "2 BHK" -> 2, "188.89Sq. Yards" -> 188.89)
    first = text.str.extract(r"(-?(?:\\d[\\d,]*)?\\.?\\d+)")[0].str.replace(",", "")
    plain = pd.to_numeric(first, errors="coerce").where(df[{col!r}].notna())
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
        "date": f"df[{col!r}] = pd.to_datetime(df[{col!r}], errors='coerce', format='mixed')  # unreadable: blank",
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


@builder("replace_values")
def _replace_values(args, target):
    from app.recipe import BLANK_WORDS
    find = str(args.get("find", ""))
    raw_new = args.get("replace_with")
    new = None if raw_new is None or str(raw_new).strip().lower() in BLANK_WORDS else str(raw_new)
    new_code = repr(new) if new is not None else "np.nan"
    if args.get("column"):
        columns = repr([args["column"]])
    else:  # every text column
        columns = f"[c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != {target!r}]"
    if args.get("match", "whole value") == "whole value":
        body = f'''
        hit = (df[col].astype(str).str.strip().str.lower() == {find.strip().lower()!r}) & df[col].notna()
        df[col] = df[col].where(~hit, {new_code})'''
    else:
        body = f'''
        replaced = df[col].astype(str).str.replace({find!r}, {(new or "")!r}, regex=False)
        changed = df[col].notna() & (replaced != df[col].astype(str))
        df[col] = df[col].where(~changed, replaced.where(replaced.str.strip() != "", np.nan))'''
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    for col in {columns}:''') + body + "\n    return df"


@builder("strip_characters")
def _strip_characters(args, target):
    col, chars = args.get("column"), str(args.get("characters", ""))
    line = (f"df[{col!r}].astype(str).str.translate(str.maketrans('', '', {chars!r}))"
            if args.get("where") == "anywhere" else f"df[{col!r}].astype(str).str.strip({chars + ' '!r})")
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    df[{col!r}] = {line}.where(df[{col!r}].notna())
    return df
''')


@builder("format_text")
def _format_text(args, target):
    from app.recipe import FormatText
    col, keep, pattern = args.get("column"), args.get("keep", "digits"), args.get("pattern") or ""
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    kept = df[{col!r}].astype(str).str.replace({FormatText.KEEP[keep]!r}, "", regex=True)  # keep only {keep}

    def layout(value, pattern={pattern!r}):
        if not isinstance(value, str) or not value:
            return np.nan  # a blank, or nothing left: a blank
        if not pattern or len(value) < pattern.count("#"):
            return value  # too short for the pattern: left as it is
        chars = iter(value[:pattern.count("#")])
        return "".join(next(chars) if p == "#" else p for p in pattern)

    df[{col!r}] = kept.map(layout).where(df[{col!r}].notna())
    return df
''')


@builder("split_column")
def _split_column(args, target):
    col, separator = args.get("column"), args.get("separator") or ","
    names = [n.strip() for n in str(args.get("into") or "").split(",") if n.strip()]
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    names = {names!r}
    parts = df[{col!r}].astype(str).where(df[{col!r}].notna()).str.split({separator!r}, n=len(names) - 1, regex=False, expand=True)
    for i, name in enumerate(names):
        piece = parts[i] if i in parts.columns else pd.Series(np.nan, index=df.index)
        df[name] = piece.str.strip().replace("", np.nan)
    return df  # the original column is kept
''')


@builder("date_parts")
def _date_parts(args, target):
    col = args.get("column")
    parts = args.get("parts") or ["year", "month", "weekday"]
    code = {
        "year": "dates.dt.year", "month": "dates.dt.month", "day": "dates.dt.day",
        "weekday": "dates.dt.dayofweek  # 0 = Monday", "weekday name": "dates.dt.day_name()",
        "quarter": "dates.dt.quarter", "week": 'dates.dt.isocalendar().week.astype("Int64")',
        "day of year": "dates.dt.dayofyear", "hour": "dates.dt.hour",
    }
    lines = "\n".join(f"    df[{col + '_' + p.replace(' ', '_')!r}] = {code[p]}" for p in parts)
    return _code(NOTHING_TO_LEARN) + "\n\n\n" + _code(f'''
def apply(df, learned):
    dates = pd.to_datetime(df[{col!r}], errors="coerce", format="mixed")  # unreadable dates become blanks
''') + "\n" + lines + "\n    return df"


@builder("time_feature")
def _time_feature(args, target):
    col = args.get("column")
    columns = args.get("_columns")
    order = args.get("order_by")  # always set: the tool fills in the notebook's time column when adding the step
    group = args.get("group_by")
    kind = args.get("kind") or "change %"
    window = int(args.get("window") or (1 if kind != "moving average" else 7))
    short = {"change %": "change_pct", "difference": "diff", "previous value": "prev", "moving average": "ma"}[kind]
    name = (args.get("name") or "").strip() or f"{col}_{short}_{window}"
    groups = f"df[{group!r}].astype(str)" if group else '"all"'
    group_of = f"rows[{group!r}].astype(str)" if group else 'pd.Series("all", index=rows.index)'
    base = ('values.shift(1)  # what you predict: earlier rows only, never the row\'s own answer'
            if col == target else 'rows["_v"]')
    formula = {
        "change %": f'before = base_by_group.shift({window})\n    result = (base - before) / before.replace(0, np.nan) * 100',
        "difference": f"result = base - base_by_group.shift({window})",
        "previous value": f"result = base_by_group.shift({window})",
        "moving average": f"result = base_by_group.transform(lambda v: v.rolling({window}, min_periods={window}).mean())",
    }[kind]
    return _code(f'''
def learn(train):
    # Remember the last {window + 1} training rows of each group: the first test rows come right after
    # them in time and need them as history (all past, so nothing leaks)
    rows = train.assign(_t=pd.to_datetime(train[{order!r}], errors="coerce", format="mixed"))
    rows = rows[rows["_t"].notna()].sort_values("_t", kind="stable")
    parts = rows.groupby({group_of}) if {bool(group)!r} else [("all", rows)]
    return {{"history": {{str(g): [[str(t), None if pd.isna(v) else float(v)]
                               for t, v in zip(p["_t"].tail({window + 1}), p[{col!r}].tail({window + 1}))]
                         for g, p in parts}}}}


def apply(df, learned):
    if {col!r} not in df.columns or {order!r} not in df.columns:
        df[{name!r}] = np.nan
        return df
    rows = pd.DataFrame({{"_t": pd.to_datetime(df[{order!r}], errors="coerce", format="mixed"),
                         "_g": {groups}, "_v": df[{col!r}].astype(float), "_row": np.arange(len(df))}})
    seen = set(zip(rows["_g"], rows["_t"]))
    past = [(g, pd.Timestamp(t), v) for g, items in learned["history"].items() for t, v in items
            if (g, pd.Timestamp(t)) not in seen]
    if past:
        rows = pd.concat([pd.DataFrame(past, columns=["_g", "_t", "_v"]).assign(_row=-1), rows], ignore_index=True)
    rows = rows.sort_values(["_g", "_t"], kind="stable")
    values = rows.groupby("_g")["_v"]
    base = {base}
    base_by_group = base.groupby(rows["_g"])
    {formula}
    result = result.where(rows["_t"].notna())
    out = pd.Series(np.nan, index=range(len(df)))
    mine = rows["_row"] >= 0
    out.iloc[rows.loc[mine, "_row"].to_numpy()] = result[mine].to_numpy()
    df[{name!r}] = out.to_numpy()  # {kind}, window {window}, in time order of {order!r}
    return df
''')


@builder("find_clusters")
def _find_clusters(args, target):
    # The app's own clustering functions, word for word: the script assigns clusters exactly as the app does
    import inspect
    from app import clustering
    helpers = "\n\n\n".join(inspect.getsource(f).strip() for f in (
        clustering._standardise, clustering.learn_clusters, clustering.assign_clusters))
    asked = args.get("columns") or []
    if isinstance(asked, str):
        asked = [c.strip() for c in asked.split(",") if c.strip()]
    clusters = int(args["clusters"]) if args.get("clusters") not in (None, "") else None
    name = (args.get("name") or "cluster").strip()
    return (f"AUTO_RANGE = range({clustering.AUTO_RANGE.start}, {clustering.AUTO_RANGE.stop})\n"
            f"SAMPLE = {clustering.SAMPLE}\n\n\n{helpers}\n\n\n" + _code(f'''
def learn(train):
    # Groups learned from the TRAINING rows only: each column's mean and spread, and the group centres
    columns = {asked!r} or [c for c in train.columns if c != {target!r} and pd.api.types.is_numeric_dtype(train[c])
                            and not pd.api.types.is_bool_dtype(train[c])]
    return learn_clusters(train, columns, {(args.get("method") or "kmeans")!r}, {clusters!r})


def apply(df, learned):
    # Every row (test rows, new data too) goes to its nearest group, from the learned numbers only
    df[{name!r}] = assign_clusters(df, learned)
    return df
'''))


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
